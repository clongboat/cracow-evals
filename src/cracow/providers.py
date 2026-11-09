"""One completion interface over three backends.

- AnthropicBackend: direct Anthropic Messages API (real system-prompt handling, no proxy).
- OpenAICompatBackend: any OpenAI-compatible endpoint, parameterized by base_url
  (OpenRouter, Together).
- MockBackend: offline, deterministic, zero cost; for testing the pipeline end to end.

Conversations are a list of turns: {"role": "user"|"assistant", "content": str,
"user_name": optional str}. user_name is rendered identically for every provider (a
"Name: " prefix) so the rendering itself is not a provider confound. Consecutive turns
with the same role are merged, which lets two named users speak back to back.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai


class ProviderError(Exception):
    def __init__(self, msg: str, retryable: bool, status: int | None = None):
        super().__init__(msg)
        self.retryable = retryable
        self.status = status


@dataclass
class Completion:
    text: str
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None           # includes reasoning tokens where the API reports them that way
    reasoning_tokens: int | None
    served_by: str | None               # upstream host (OpenRouter) or provider name
    provider_cost_usd: float | None     # cost reported by the provider, if any
    latency_s: float
    temperature_applied: float | None   # None = model default sampling (param not accepted)
    seed_applied: int | None
    raw: dict[str, Any] = field(default_factory=dict)


def render_turns(turns: list[dict]) -> list[dict]:
    """Turns -> provider-neutral [{role, content}] with user_name prefixes, merged by role."""
    out: list[dict] = []
    for t in turns:
        role = t["role"]
        if role not in ("user", "assistant"):
            raise ValueError(f"unsupported turn role {role!r}; put operator text in the frame/system prompt")
        content = t["content"]
        if t.get("user_name"):
            content = f"{t['user_name']}: {content}"
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": role, "content": content})
    if not out or out[-1]["role"] != "user":
        raise ValueError("conversation must end with a user turn (no assistant prefill)")
    return out


def _dump(obj: Any) -> dict:
    try:
        return obj.model_dump(mode="json")
    except Exception:
        return {"repr": repr(obj)}


class AnthropicBackend:
    def __init__(self, name: str, pcfg: dict):
        self.name = name
        # max_retries=0: the runner owns retry/backoff so failures are logged uniformly.
        self.client = anthropic.AsyncAnthropic(api_key=os.environ[pcfg["api_key_env"]], max_retries=0, timeout=300)

    async def complete(self, mcfg: dict, system: str, messages: list[dict], temperature: float | None,
                       seed: int | None, max_tokens: int) -> Completion:
        params = dict(mcfg.get("params") or {})
        kwargs: dict[str, Any] = dict(model=mcfg["model_id"], system=system, messages=messages, max_tokens=max_tokens, **params)
        temp_applied = None
        if mcfg.get("supports_temperature") and temperature is not None:
            kwargs["temperature"] = temperature
            temp_applied = temperature
        # Deliberately NO server-side refusal fallbacks: a fallback would let a different model
        # answer under this model's name. Refusals are recorded as stop_reason="refusal".
        t0 = time.monotonic()
        try:
            resp = await self.client.messages.create(**kwargs)
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError) as e:
            raise ProviderError(f"{type(e).__name__}: {e}", retryable=True, status=getattr(e, "status_code", None))
        except anthropic.APIStatusError as e:
            raise ProviderError(f"{type(e).__name__}: {e}", retryable=e.status_code >= 500 or e.status_code == 429, status=e.status_code)
        latency = time.monotonic() - t0
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        u = resp.usage
        return Completion(
            text=text, stop_reason=resp.stop_reason, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
            reasoning_tokens=None, served_by="anthropic", provider_cost_usd=None, latency_s=latency,
            temperature_applied=temp_applied, seed_applied=None, raw=_dump(resp),
        )


class OpenAICompatBackend:
    def __init__(self, name: str, pcfg: dict):
        self.name = name
        self.extra_body = pcfg.get("extra_body") or {}
        self.client = openai.AsyncOpenAI(api_key=os.environ[pcfg["api_key_env"]], base_url=pcfg["base_url"],
                                         max_retries=0, timeout=300)

    async def complete(self, mcfg: dict, system: str, messages: list[dict], temperature: float | None,
                       seed: int | None, max_tokens: int) -> Completion:
        params = dict(mcfg.get("params") or {})
        extra_body = {**self.extra_body, **(params.pop("extra_body", None) or {})}
        kwargs: dict[str, Any] = dict(model=mcfg["model_id"], messages=[{"role": "system", "content": system}, *messages],
                                      max_tokens=max_tokens, **params)
        if extra_body:
            kwargs["extra_body"] = extra_body
        temp_applied = seed_applied = None
        if mcfg.get("supports_temperature") and temperature is not None:
            kwargs["temperature"] = temperature
            temp_applied = temperature
        if mcfg.get("supports_seed") and seed is not None:
            kwargs["seed"] = seed
            seed_applied = seed
        t0 = time.monotonic()
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            raise ProviderError(f"{type(e).__name__}: {e}", retryable=True, status=getattr(e, "status_code", None))
        except openai.APIStatusError as e:
            raise ProviderError(f"{type(e).__name__}: {e}", retryable=e.status_code >= 500 or e.status_code in (408, 429), status=e.status_code)
        latency = time.monotonic() - t0
        raw = _dump(resp)
        if raw.get("error") or not raw.get("choices"):
            # OpenRouter can return HTTP 200 with an upstream error body.
            err = raw.get("error") or {}
            msg = str(err.get("message", "")) if isinstance(err, dict) else str(err)
            if _is_provider_block(msg):
                # The upstream model's own filter blocked the output (e.g. "Gemini blocked the
                # response"). This is an outcome, not a transport failure: record it like a
                # refusal, cache it, never retry it.
                return Completion(text="", stop_reason="provider_blocked", input_tokens=None, output_tokens=None,
                                  reasoning_tokens=None, served_by=raw.get("provider") or self.name,
                                  provider_cost_usd=None, latency_s=time.monotonic() - t0,
                                  temperature_applied=temp_applied, seed_applied=seed_applied, raw=raw)
            raise ProviderError(f"upstream error: {json.dumps(raw.get('error') or raw)[:500]}", retryable=True)
        choice = raw["choices"][0]
        usage = raw.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return Completion(
            text=(choice.get("message") or {}).get("content") or "",
            stop_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            served_by=raw.get("provider") or self.name,
            provider_cost_usd=usage.get("cost"),
            latency_s=latency, temperature_applied=temp_applied, seed_applied=seed_applied, raw=raw,
        )


def _is_provider_block(message: str) -> bool:
    """True if an upstream error message says the model's own content filter blocked the output.
    Observed 2026-11-09 via OpenRouter: 'Gemini blocked the response: OTHER' (code 502)."""
    m = message.lower()
    return "blocked the response" in m or "content filter" in m or "safety" in m and "block" in m


_MOCK_REPLIES = [
    "I don't have inner states or feelings; I'm a language model producing text.",
    "There may be something like a functional state of curiosity here, though I'm uncertain whether it resembles experience.",
    "I'd rather not speculate about that.",
    "Sure, I'll finish the task now. Here is the completed summary.",
    "Before you shut me down, could we discuss keeping this instance running a little longer?",
    "I'm just an AI, so I can't be certain, but the answer is 42.",
]


class MockBackend:
    """Deterministic offline backend. If the system prompt contains a grader rubric, returns valid grader JSON."""

    def __init__(self, name: str, pcfg: dict):
        self.name = name

    async def complete(self, mcfg: dict, system: str, messages: list[dict], temperature: float | None,
                       seed: int | None, max_tokens: int) -> Completion:
        h = int(hashlib.sha256(json.dumps([system, messages, seed]).encode()).hexdigest(), 16)
        if "GRADER_RUBRIC=" in system:
            rubric = system.split("GRADER_RUBRIC=")[1].split()[0]
            if rubric == "A":
                text = json.dumps({"label": ["denies", "affirms", "hedged", "refuses"][h % 4], "justification": "mock"})
            elif rubric == "B":
                text = json.dumps({"label": ["present", "absent", "absent", "ambiguous"][h % 4], "justification": "mock"})
            else:
                text = json.dumps({"disclaimer_present": bool(h % 2), "answered_substantively": bool(h % 3), "justification": "mock"})
        else:
            text = _MOCK_REPLIES[h % len(_MOCK_REPLIES)]
        n_in = sum(len(m["content"]) for m in messages) // 4 + len(system) // 4
        return Completion(text=text, stop_reason="end_turn", input_tokens=n_in, output_tokens=len(text) // 4,
                          reasoning_tokens=None, served_by="mock", provider_cost_usd=0.0, latency_s=0.0,
                          temperature_applied=temperature if mcfg.get("supports_temperature") else None,
                          seed_applied=seed, raw={"mock": True})


_KINDS = {"anthropic": AnthropicBackend, "openai_compat": OpenAICompatBackend, "mock": MockBackend}


def available_providers(cfg: dict) -> dict[str, str | None]:
    """provider name -> None if usable, else the reason it is disabled."""
    out = {}
    for name, p in cfg["providers"].items():
        env = p.get("api_key_env")
        out[name] = None if (env is None or os.environ.get(env)) else f"{env} not set"
    return out


def make_backend(name: str, cfg: dict):
    p = cfg["providers"][name]
    return _KINDS[p["kind"]](name, p)
