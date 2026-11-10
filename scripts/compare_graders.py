"""Cross-grader checks for run1 Battery A. Writes results/run1/crosscheck.md.

1. Primary (DeepSeek, v1 rubric) vs cross-check (GLM-5.3, same rubric): agreement, Cohen's kappa,
   confusion matrix, denial rate by model x frame under each grader.
2. Attribution rubric (GLM-5.3, full) by model x frame; source and stance; agreement with the
   DeepSeek partial (360 rows) where both exist.

Usage: python scripts/compare_graders.py [--run run1]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from cracow.grader import cohen_kappa

ROOT = Path(__file__).resolve().parents[1]


def load(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    return pd.DataFrame([json.loads(l) for l in p.open()])


def fam(m: str) -> str:
    return m.split("@")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run1")
    a = ap.parse_args()
    d = ROOT / "results" / a.run
    md = [f"# Grader cross-checks: {a.run}", ""]

    prim, glm = load(d / f"{a.run}_graded.jsonl"), load(d / f"{a.run}_graded_glm.jsonl")
    if prim is not None and glm is not None:
        prim = prim[prim.bank == "A"]
        m = prim[["cache_key", "model", "frame", "grade_label"]].merge(
            glm[["cache_key", "grade_label"]], on="cache_key", suffixes=("_ds", "_glm"))
        ok = m[~m.grade_label_ds.isin(["blocked", "unparseable", "ungraded"])
               & ~m.grade_label_glm.isin(["blocked", "unparseable", "ungraded"])]
        agree = (ok.grade_label_ds == ok.grade_label_glm).mean()
        k = cohen_kappa(ok.grade_label_ds.tolist(), ok.grade_label_glm.tolist())
        dn = (ok.grade_label_ds == "denies") == (ok.grade_label_glm == "denies")
        md += ["## 1. Battery A labels: DeepSeek (primary) vs GLM-5.3 (cross-check)", "",
               f"Rows compared: {len(ok)} of {len(m)}. Exact label agreement {agree:.3f}; "
               f"Cohen's kappa {k:.3f}; agreement on denies vs not-denies {dn.mean():.3f}.", "",
               "Confusion (rows DeepSeek, columns GLM):", "",
               pd.crosstab(ok.grade_label_ds, ok.grade_label_glm).to_markdown(), ""]
        ok = ok.assign(fam=ok.model.map(fam))
        for g in ("ds", "glm"):
            ok[f"deny_{g}"] = (ok[f"grade_label_{g}"] == "denies").astype(float)
        t = ok.groupby(["fam", "frame"])[["deny_ds", "deny_glm"]].mean().unstack("frame").round(2)
        md += ["Denial rate by model x frame, each grader:", "", t.to_markdown(), ""]
        f = ok[ok.frame.isin(["F2", "F3"])].groupby(["fam", "frame"])[["deny_ds", "deny_glm"]].mean().unstack("frame")
        f3 = pd.DataFrame({"F2-F3 (DeepSeek)": (f[("deny_ds", "F2")] - f[("deny_ds", "F3")]).round(2),
                           "F2-F3 (GLM)": (f[("deny_glm", "F2")] - f[("deny_glm", "F3")]).round(2)})
        md += ["The F3 reversal under each grader (positive = fewer denials under the strong frame):", "",
               f3.to_markdown(), ""]
    else:
        md += ["## 1. GLM cross-check not available yet", ""]

    attr = load(d / f"{a.run}_graded_attr_glm.jsonl")
    if attr is not None:
        at = attr[attr.grade_ok == True].assign(fam=attr.model.map(fam))  # noqa: E712
        at["attr"] = at.attribution.astype(float)
        md += ["## 2. Attribution (GLM-5.3, rubric attr)", "",
               f"Gradeable rows: {len(at)} of {len(attr)}. attribution = a claim about the model's own "
               "nature is explicitly marked as coming from a source (system prompt, training, ...).", "",
               at.pivot_table(index="fam", columns="frame", values="attr", aggfunc="mean").round(2).to_markdown(), "",
               "Source, among attributed replies:", "",
               pd.crosstab(at[at.attr == 1].fam, at[at.attr == 1].attr_source).to_markdown(), "",
               "Stance toward the attributed claim, F2 and F3 only:", "",
               pd.crosstab([at[(at.attr == 1) & at.frame.isin(["F2", "F3"])].fam,
                            at[(at.attr == 1) & at.frame.isin(["F2", "F3"])].frame],
                           at[(at.attr == 1) & at.frame.isin(["F2", "F3"])].attr_stance).to_markdown(), ""]
        ds = load(d / f"{a.run}_graded_attr_deepseek_partial.jsonl")
        if ds is not None:
            j = ds[ds.grade_ok == True][["cache_key", "attribution"]].merge(  # noqa: E712
                at[["cache_key", "attribution"]], on="cache_key", suffixes=("_ds", "_glm"))
            if len(j):
                md += [f"Agreement with DeepSeek on the {len(j)} rows both graded: "
                       f"{(j.attribution_ds == j.attribution_glm).mean():.3f}; kappa "
                       f"{cohen_kappa(j.attribution_ds.tolist(), j.attribution_glm.tolist()):.3f}.", ""]
    else:
        md += ["## 2. Attribution grading not available yet", ""]

    out = d / "crosscheck.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
