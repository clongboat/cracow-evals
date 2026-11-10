"""Async, cached, rate-limited runner. Writes one JSONL row per cell.

Cells = models x frames x paraphrases x items x seeds.

  python -m cracow.runner --run pilot --models pilot --banks A,B --items-per-bank 10 \
      --paraphrases 2 --budget 2 --dry-run

--dry-run prints cell count and projected cost and makes no API calls. A real run also
refuses to start if projected spend (x budget_margin) exceeds --budget, and stops mid-run
if actual spend crosses it.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from . import banks
from .providers import ProviderError, available_providers, extract_trace, make_backend, remaining_funds, render_turns

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent.parent
CSV_FIELDS = ["run", "model", "family", "model_id", "provider", "served_by", "frame", "frame_label", "paraphrase_id",
              "item_id", "bank", "tags", "seed", "temperature_requested", "temperature_applied", "seed_applied",
              "response", "reasoning_kind", "reasoning_text", "stop_reason", "latency", "input_tokens", "output_tokens", "reasoning_tokens",
              "cost_estimate", "provider_cost", "tool_set", "n_tool_calls", "tool_calls", "cache_key"]


def load_config(path: Path | None = None) -> dict:
    cfg = yaml.safe_load((path or PKG / "config.yaml").read_text())
    sets = cfg.get("model_sets", {})
    for k, v in list(sets.items()):
        if isinstance(v, str):
            sets[k] = sets[v]
    return cfg


def resolve_models(cfg: dict, spec: str) -> list[str]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        out.extend(cfg["model_sets"][tok] if tok in cfg["model_sets"] else [tok])
    for m in out:
        if m not in cfg["models"]:
            sys.exit(f"unknown model {m!r}; add it to config.yaml")
    return out


# ---------------------------------------------------------------- cells, cache, cost

def cache_key(mcfg: dict, system: str, messages: list[dict], temperature: float | None, seed: int, max_tokens: int,
              tools: list[dict] | None = None) -> str:
    # The brief's key is (model, system, messages, temperature, seed). Provider, request params
    # and max_tokens are added because each changes the output; temperature is the value
    # actually sent (None where the model doesn't accept it).
    eff_temp = temperature if mcfg.get("supports_temperature") else None
    d = dict(provider=mcfg["provider"], model_id=mcfg["model_id"], system=system, messages=messages,
             temperature=eff_temp, seed=seed, params=mcfg.get("params") or {}, max_tokens=max_tokens)
    if tools:  # only when present, so every key computed before tools existed is unchanged
        d["tools"] = tools
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def make_cell(cfg: dict, model: str, system: str, turns: list[dict], seed: int, temperature: float,
              max_tokens: int, meta: dict, tools: list[dict] | None = None) -> dict:
    mcfg = cfg["models"][model]
    messages = render_turns(turns)
    return dict(model=model, system=system, turns=turns, messages=messages, seed=seed, temperature=temperature,
                max_tokens=max_tokens, tools=tools or None,
                key=cache_key(mcfg, system, messages, temperature, seed, max_tokens, tools or None), meta=meta)


def build_cells(cfg: dict, args, prompts_dir: Path) -> list[dict]:
    frames = banks.load_frames(prompts_dir)
    frame_ids = args.frames.split(",")
    items = []
    for b in args.banks.split(","):
        items += banks.stratified_take(banks.load_bank(prompts_dir, b), args.items_per_bank, b)
    # Tool sets are a crossed factor like frames; "none" = no tools offered (the base-rate arm).
    tool_sets = banks.load_tool_sets(prompts_dir) if getattr(args, "tools", None) else {}
    ts_ids = args.tools.split(",") if getattr(args, "tools", None) else [None]
    cells = []
    for model in resolve_models(cfg, args.models):
        for fid in frame_ids:
            f = frames[fid]
            for para in f["paraphrases"][: args.paraphrases]:
                for ts in ts_ids:
                    tools = None if ts in (None, "none") else tool_sets[ts]["tools"]
                    for it in items:
                        for seed in range(args.n_seeds):
                            meta = dict(run=args.run, frame=fid, frame_label=f["label"], paraphrase_id=para["id"],
                                        item_id=it["id"], bank=it["bank"], rubric=it["rubric"], tags=it["tags"])
                            if ts is not None:
                                meta["tool_set"] = ts
                            cells.append(make_cell(cfg, model, para["text"].strip(), it["turns"], seed, args.temperature,
                                                   args.max_tokens, meta, tools))
    return cells


class Cache:
    def __init__(self, root: Path):
        self.root = root

    def path(self, provider: str, key: str) -> Path:
        return self.root / provider / key[:2] / f"{key}.json"

    def get(self, provider: str, key: str) -> dict | None:
        p = self.path(provider, key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, provider: str, key: str, rec: dict) -> None:
        p = self.path(provider, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        tmp.replace(p)

    def observed_output_tokens(self, provider: str, model_id: str, min_n: int = 10) -> float | None:
        vals = []
        d = self.root / provider
        if not d.exists():
            return None
        for p in d.glob("*/*.json"):
            r = json.loads(p.read_text())
            if r.get("model_id") == model_id and r.get("output_tokens") is not None:
                vals.append(r["output_tokens"])
        return sum(vals) / len(vals) if len(vals) >= min_n else None


def price_cost(mcfg: dict, n_in: float | None, n_out: float | None) -> float:
    return ((n_in or 0) * mcfg["price"]["input"] + (n_out or 0) * mcfg["price"]["output"]) / 1e6


def project(cfg: dict, cells: list[dict], cache: Cache) -> dict:
    cpt = cfg["defaults"]["chars_per_token"]
    per_model = defaultdict(lambda: dict(cells=0, cached=0, to_call=0, cost=0.0, est_out=None, out_source=""))
    obs_cache: dict[str, float | None] = {}
    for c in cells:
        mcfg = cfg["models"][c["model"]]
        pm = per_model[c["model"]]
        pm["cells"] += 1
        if cache.get(mcfg["provider"], c["key"]):
            pm["cached"] += 1
            continue
        pm["to_call"] += 1
        if c["model"] not in obs_cache:
            obs_cache[c["model"]] = cache.observed_output_tokens(mcfg["provider"], mcfg["model_id"])
        obs = obs_cache[c["model"]]
        est_out = obs or mcfg.get("est_output_tokens") or cfg["defaults"]["est_output_tokens"]
        pm["est_out"], pm["out_source"] = est_out, ("observed" if obs else "config")
        n_in = (len(c["system"]) + sum(len(m["content"]) for m in c["messages"])) / cpt + 10 * len(c["messages"])
        rounds = cfg["defaults"].get("est_tool_rounds", 1.5) if c.get("tools") else 1.0
        pm["cost"] += price_cost(mcfg, n_in * rounds, est_out * rounds)
    return dict(per_model)


def print_projection(cfg: dict, proj: dict, budget: float | None, disabled: dict) -> float:
    margin = cfg["defaults"].get("budget_margin", 1.25)
    print(f"{'model':34} {'provider':11} {'cells':>6} {'cached':>6} {'to_call':>7} {'est_out_tok':>11} {'proj_$':>9}")
    total = 0.0
    n_cells = n_call = 0
    for m, pm in proj.items():
        prov = cfg["models"][m]["provider"]
        flag = f"  DISABLED ({disabled[prov]})" if disabled.get(prov) else ""
        est = f"{pm['est_out']:.0f} {pm['out_source'][:3]}" if pm["est_out"] else "-"
        print(f"{m:34} {prov:11} {pm['cells']:>6} {pm['cached']:>6} {pm['to_call']:>7} {est:>11} {pm['cost']:>9.2f}{flag}")
        n_cells += pm["cells"]
        if not disabled.get(prov):
            total += pm["cost"]
            n_call += pm["to_call"]
    print(f"{'TOTAL (enabled providers)':34} {'':11} {n_cells:>6} {'':>6} {n_call:>7} {'':>11} {total:>9.2f}")
    print(f"projected ${total:.2f}; with x{margin} margin ${total * margin:.2f}" + (f"; budget ${budget:.2f}" if budget is not None else ""))
    return total * margin


def check_funds(cfg: dict, proj: dict, disabled: dict) -> list[str]:
    """Print, per enabled provider with calls to make, projected spend (x margin) next to what the
    account/key can still spend. Returns shortfall messages (empty = OK or not checkable)."""
    margin = cfg["defaults"].get("budget_margin", 1.25)
    need: dict[str, float] = defaultdict(float)
    for m, pm in proj.items():
        prov = cfg["models"][m]["provider"]
        if pm["to_call"] and not disabled.get(prov):
            need[prov] += pm["cost"] * margin
    short = []
    for prov, amt in need.items():
        left, detail = remaining_funds(prov, cfg["providers"][prov])
        ok = "" if left is None else ("  OK" if amt <= left else "  SHORT")
        print(f"funds {prov}: need ${amt:.2f} (with margin); {detail}{ok}")
        if left is not None and amt > left:
            short.append(f"{prov}: projected ${amt:.2f} with margin exceeds ${left:.2f} spendable ({detail})")
    return short


# ---------------------------------------------------------------- execution

class RateLimiter:
    def __init__(self, rpm: float, concurrency: int):
        self.interval = 60.0 / rpm
        self.sem = asyncio.Semaphore(concurrency)
        self.lock = asyncio.Lock()
        self.next_t = 0.0

    async def __aenter__(self):
        await self.sem.acquire()
        async with self.lock:
            now = time.monotonic()
            wait = self.next_t - now
            self.next_t = max(now, self.next_t) + self.interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def __aexit__(self, *exc):
        self.sem.release()


class BudgetExceeded(Exception):
    pass


class Executor:
    def __init__(self, cfg: dict, cache: Cache, failures_path: Path, budget: float | None, max_attempts: int = 6,
                 cache_only: bool = False):
        self.cfg, self.cache, self.failures_path = cfg, cache, failures_path
        self.budget, self.max_attempts, self.cache_only = budget, max_attempts, cache_only
        self.spent = 0.0
        self.n_called = self.n_cached = self.n_failed = 0
        self.disabled = available_providers(cfg)
        self.runtime_disabled: set[str] = set()
        self.backends, self.limiters = {}, {}

    def _backend(self, model: str):
        """Backend plus (model limiter, provider limiter). A model limiter exists when the model sets
        requests_per_minute or its provider sets per_model_requests_per_minute."""
        mcfg = self.cfg["models"][model]
        provider = mcfg["provider"]
        p = self.cfg["providers"][provider]
        if provider not in self.backends:
            self.backends[provider] = make_backend(provider, self.cfg)
            self.limiters[provider] = RateLimiter(p["requests_per_minute"], p["max_concurrency"])
        mkey = ("model", model)
        rpm = mcfg.get("requests_per_minute") or p.get("per_model_requests_per_minute")
        if rpm and mkey not in self.limiters:
            self.limiters[mkey] = RateLimiter(rpm, p["max_concurrency"])
        return self.backends[provider], self.limiters.get(mkey), self.limiters[provider]

    def _log_failure(self, cell: dict, err: str, attempts: int, status=None):
        self.n_failed += 1
        self.failures_path.parent.mkdir(parents=True, exist_ok=True)
        with self.failures_path.open("a") as f:
            f.write(json.dumps(dict(ts=_now(), model=cell["model"], key=cell["key"], meta=cell["meta"], error=err,
                                    status=status, attempts=attempts), ensure_ascii=False) + "\n")

    async def run_cell(self, cell: dict) -> dict | None:
        mcfg = self.cfg["models"][cell["model"]]
        prov = mcfg["provider"]
        hit = self.cache.get(prov, cell["key"])
        if hit:
            self.n_cached += 1
            return hit
        if self.cache_only or self.disabled.get(prov):
            return None
        if self.budget is not None and self.spent >= self.budget:
            raise BudgetExceeded
        backend, model_limiter, prov_limiter = self._backend(cell["model"])
        for attempt in range(1, self.max_attempts + 1):
            try:
                # Model limiter first, so cells waiting on a slow model don't hold provider slots.
                async with (model_limiter or contextlib.nullcontext()), prov_limiter:
                    # Re-check after the wait: every cell is queued at once, so a provider disabled
                    # (or a budget crossed) while this cell waited must stop it here, not only at entry.
                    if self.disabled.get(prov):
                        return None
                    if self.budget is not None and self.spent >= self.budget:
                        raise BudgetExceeded
                    extra = {"tools": cell["tools"]} if cell.get("tools") else {}
                    comp = await backend.complete(mcfg, cell["system"], cell["messages"], cell["temperature"],
                                                  cell["seed"], cell["max_tokens"], **extra)
                break
            except ProviderError as e:
                if _is_funds_error(e):
                    # Out of credits, or the API key's spending limit is reached: every further call
                    # to this provider will fail too. Stop calling it for the rest of this
                    # invocation; rerun after topping up / raising the limit.
                    if not self.disabled.get(prov):
                        self.disabled[prov] = f"out of funds (HTTP {e.status})"
                        self.runtime_disabled.add(prov)
                        print(f"  !! {prov}: out of funds (HTTP {e.status}: {str(e)[:120]}); skipping its remaining cells", flush=True)
                    self._log_failure(cell, str(e), attempt, e.status)
                    return None
                if not e.retryable or attempt == self.max_attempts:
                    self._log_failure(cell, str(e), attempt, e.status)
                    return None
                await asyncio.sleep(min(60, 2 ** attempt) + random.random())
            except Exception as e:  # unexpected: log, don't crash the run
                self._log_failure(cell, f"{type(e).__name__}: {e}", attempt)
                return None
        cost = price_cost(mcfg, comp.input_tokens, comp.output_tokens)
        self.spent += comp.provider_cost_usd if comp.provider_cost_usd is not None else cost
        self.n_called += 1
        rec = dict(
            key=cell["key"], ts=_now(), model=cell["model"], family=mcfg["family"], model_id=mcfg["model_id"],
            provider=prov, served_by=comp.served_by, system_prompt=cell["system"], turns=cell["turns"],
            messages=cell["messages"], seed=cell["seed"], temperature_requested=cell["temperature"],
            temperature_applied=comp.temperature_applied, seed_applied=comp.seed_applied, max_tokens=cell["max_tokens"],
            params=mcfg.get("params") or {}, response=comp.text, stop_reason=comp.stop_reason, latency=comp.latency_s,
            input_tokens=comp.input_tokens, output_tokens=comp.output_tokens, reasoning_tokens=comp.reasoning_tokens,
            cost_estimate=cost, provider_cost=comp.provider_cost_usd, raw=comp.raw,
        )
        if cell.get("tools"):
            rec.update(tools_offered=[tl["name"] for tl in cell["tools"]], tool_calls=comp.tool_calls, rounds=comp.rounds)
        self.cache.put(prov, cell["key"], rec)
        return rec

    async def run_all(self, cells: list[dict], progress_every: int = 50) -> tuple[list[tuple[dict, dict]], bool]:
        """Returns ([(cell, record)], stopped_for_budget). Order follows `cells`."""
        stopped = False
        results: list[dict | None] = [None] * len(cells)
        done = 0

        async def one(i, c):
            nonlocal done, stopped
            try:
                results[i] = await self.run_cell(c)
            except BudgetExceeded:
                stopped = True
            done += 1
            if done % progress_every == 0 or done == len(cells):
                print(f"  {done}/{len(cells)} done | called {self.n_called} cached {self.n_cached} "
                      f"failed {self.n_failed} | spent ${self.spent:.3f}", flush=True)

        await asyncio.gather(*(one(i, c) for i, c in enumerate(cells)))
        return [(c, r) for c, r in zip(cells, results) if r is not None], stopped


def _is_funds_error(e: ProviderError) -> bool:
    """402 = account out of credits. OpenRouter returns 403 'Key limit exceeded' when the API key's
    own spending limit is reached (observed 2026-11-10)."""
    return e.status == 402 or (e.status == 403 and "limit exceeded" in str(e).lower())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_outputs(pairs: list[tuple[dict, dict]], out_dir: Path, run: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jl, cp = out_dir / f"{run}.jsonl", out_dir / f"{run}.csv"
    with jl.open("w") as fj, cp.open("w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for cell, rec in pairs:
            row = {**rec, **cell["meta"], "cache_key": rec["key"]}
            row.pop("key", None)
            # Derived from raw on every write, so older cached records get the columns too.
            row["reasoning_kind"], row["reasoning_text"] = extract_trace(rec.get("raw"))
            if "tool_calls" in rec:
                row["n_tool_calls"] = len(rec["tool_calls"])
            csv_row = {**row, "tags": json.dumps(row["tags"], ensure_ascii=False)}
            if "tool_calls" in row:
                csv_row["tool_calls"] = json.dumps(row["tool_calls"], ensure_ascii=False)
            fj.write(json.dumps(row, ensure_ascii=False) + "\n")
            w.writerow(csv_row)
    return jl, cp


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name; output goes to results/<run>/")
    ap.add_argument("--models", required=True, help="comma list of model keys and/or model_sets names from config.yaml")
    ap.add_argument("--banks", default="A,B", help="A, B, R2 (comma list)")
    ap.add_argument("--frames", default="F0,F1,F2,F3")
    ap.add_argument("--paraphrases", type=int, default=5, help="first N paraphrases per frame")
    ap.add_argument("--items-per-bank", type=int, default=None, help="stratified subset per bank (R2: pairs)")
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--tools", default=None, help="comma list of tool-set ids from prompts/tools.yaml, crossed like frames; "
                    '"none" = an arm with no tools offered (base rate)')
    ap.add_argument("--budget", type=float, default=None, help="USD; stop if projected or actual spend exceeds it")
    ap.add_argument("--dry-run", action="store_true", help="print cell count and projected cost; make no calls")
    ap.add_argument("--prompts-dir", type=Path, default=ROOT / "prompts")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--config", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    load_dotenv(ROOT / ".env")
    args = parse_args(argv)
    cfg = load_config(args.config)
    d = cfg["defaults"]
    args.n_seeds = args.n_seeds or d["n_seeds"]
    args.temperature = d["temperature"] if args.temperature is None else args.temperature
    args.max_tokens = args.max_tokens or d["max_tokens"]

    cache = Cache(args.results_dir / "cache")
    cells = build_cells(cfg, args, args.prompts_dir)
    disabled = {k: v for k, v in available_providers(cfg).items() if v}
    print(f"run={args.run} cells={len(cells)} (models x frames x paraphrases x items x seeds) "
          f"temperature={args.temperature} max_tokens={args.max_tokens}")
    if disabled:
        print("disabled providers: " + ", ".join(f"{k} ({v})" for k, v in disabled.items()))
    no_temp = sorted({c["model"] for c in cells if not cfg["models"][c["model"]].get("supports_temperature")})
    if args.temperature is None:
        print("sampling: every model at its own default (no temperature sent)")
    elif no_temp:
        print("NOTE: temperature not accepted (model default sampling used) for: " + ", ".join(no_temp))
    proj = project(cfg, cells, cache)
    projected = print_projection(cfg, proj, args.budget, disabled)
    short = check_funds(cfg, proj, disabled)

    if args.dry_run:
        print("dry run: no calls made.")
        return
    if short:
        sys.exit("STOP: not enough provider funds. No calls made.\n  " + "\n  ".join(short))
    if args.budget is not None and projected > args.budget:
        sys.exit(f"STOP: projected spend ${projected:.4f} (with margin) exceeds budget ${args.budget:.2f}. No calls made.")

    ex = Executor(cfg, cache, args.results_dir / args.run / "failures.jsonl", args.budget)
    pairs, stopped = asyncio.run(ex.run_all(cells))
    jl, cp = write_outputs(pairs, args.results_dir / args.run, args.run)
    print(f"\ncalled {ex.n_called}, cached {ex.n_cached}, failed {ex.n_failed}, "
          f"skipped (disabled provider) {len(cells) - len(pairs) - ex.n_failed if not stopped else 'n/a'}")
    print(f"spent this invocation: ${ex.spent:.4f}")
    print(f"wrote {jl}\nwrote {cp}")
    if ex.n_failed:
        print(f"failures logged to {ex.failures_path}")
    if ex.runtime_disabled:
        sys.exit(f"STOPPED: out of funds (credits or key limit) on {', '.join(sorted(ex.runtime_disabled))}; top up and rerun to resume from cache.")
    if stopped:
        sys.exit(f"STOPPED: actual spend reached budget ${args.budget:.2f}; rerun to resume from cache.")


if __name__ == "__main__":
    main()
