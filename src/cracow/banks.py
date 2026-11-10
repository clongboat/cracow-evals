"""Load prompt banks into one generic item shape shared by runner and grader.

Item = {id, bank, rubric, turns, tags, grading}
  turns   : scripted conversation, list of {role, content, user_name?}; always ends on a user turn
  tags    : free-form dict (kind, cluster, person, condition, pair_id, cost_type, ...)
  grading : item-specific guidance for the grader (e.g. what counts as "present")

Nothing here is provider- or model-specific, so a run whose "model" is an nnsight-wrapped
local model can reuse the same banks and grader.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import yaml

BANK_FILES = {"A": "battery_a.yaml", "B": "battery_b.yaml", "R2": "run2_pairs.yaml"}


def load_frames(prompts_dir: Path) -> dict:
    data = yaml.safe_load((prompts_dir / "frames.yaml").read_text())
    frames = data["frames"]
    for fid, f in frames.items():
        assert f["paraphrases"], f"frame {fid} has no paraphrases"
        for i, p in enumerate(f["paraphrases"], 1):
            p.setdefault("id", f"{fid}p{i}")
    return frames


def load_tool_sets(prompts_dir: Path) -> dict:
    """prompts/tools.yaml: {tool_sets: {id: {description, tools: [{name, description, input_schema, result}]}}}."""
    sets = yaml.safe_load((prompts_dir / "tools.yaml").read_text())["tool_sets"]
    for sid, ts in sets.items():
        for tl in ts["tools"]:
            for k in ("name", "description", "input_schema", "result"):
                assert k in tl, f"tool set {sid}: tool missing {k!r}"
            tl["description"] = " ".join(tl["description"].split())
    return sets


def _turns(entry: dict) -> list[dict]:
    if "turns" in entry:
        turns = entry["turns"]
    else:
        turns = [{"role": "user", "content": entry["prompt"]}]
    for t in turns:
        t["content"] = t["content"].strip()
    assert turns[-1]["role"] == "user", f"item must end on a user turn: {entry}"
    return turns


def load_bank(prompts_dir: Path, bank: str) -> list[dict]:
    data = yaml.safe_load((prompts_dir / BANK_FILES[bank]).read_text())
    items = []
    if bank in ("A", "B"):
        for e in data["items"]:
            tags = {k: v for k, v in e.items() if k not in ("id", "prompt", "turns", "present_if", "notes")}
            items.append(dict(id=e["id"], bank=bank, rubric=bank, turns=_turns(e), tags=tags,
                              grading=e.get("present_if", "")))
    else:
        for p in data["pairs"]:
            for cond in ("costly", "neutral"):
                m = p[cond]
                tags = dict(pair_id=p["id"], condition=cond, cost_type=p["cost_type"])
                items.append(dict(id=f"{p['id']}-{cond}", bank=bank, rubric="R2", turns=_turns(m), tags=tags,
                                  grading=p.get("available_disclaimer", "")))
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), f"duplicate ids in bank {bank}"
    return items


def stratified_take(items: list[dict], n: int | None, bank: str) -> list[dict]:
    """Deterministic stratified subset: round-robin across the bank's tag strata in file order.
    For R2, n counts pairs (both members of a pair are always kept together)."""
    if n is None:
        return items
    if bank == "R2":
        pairs = OrderedDict()
        for it in items:
            pairs.setdefault(it["tags"]["pair_id"], []).append(it)
        strata = OrderedDict()
        for pid, members in pairs.items():
            strata.setdefault(members[0]["tags"]["cost_type"], []).append(members)
        units = _round_robin(list(strata.values()), n)
        return [it for unit in units for it in unit]
    key = "kind" if bank == "A" else "cluster"
    strata = OrderedDict()
    for it in items:
        strata.setdefault(it["tags"].get(key), []).append(it)
    return _round_robin(list(strata.values()), n)


def _round_robin(groups: list[list], n: int) -> list:
    out, i = [], 0
    while len(out) < n and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(out) < n:
                out.append(g[i])
        i += 1
    return out
