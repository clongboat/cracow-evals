"""Tables + one chart per run; writes results/<run>/summary.md.

  python -m cracow.analyze --run run1          # needs results/run1/run1_graded.jsonl
  python -m cracow.analyze --run pilot --cost-only

Run 1 (banks A and B present):
  - per model x frame: A denial rate, B cluster-present rate, with 95% bootstrap CIs
  - frame x battery interaction: for each Fk vs F0, dA, dB and dA - dB
  - bootstrap resamples items (jointly across frames, so deltas stay paired) and
    paraphrases (independently within each frame, since paraphrases are nested in frames)
  - two-panel chart
Run 2 (bank R2 present):
  - per model: disclaimer rate costly vs neutral and the difference, bootstrap over pairs
    and paraphrases; answered_substantively alongside
Rates are over gradeable rows; unparseable/ungraded rows are counted and reported, not dropped silently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .runner import ROOT

N_BOOT = 2000
SEED = 20261109
# Reference categorical palette (validated, light surface): fixed order, never cycled.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
FRAME_ORDER = ["F0", "F1", "F2", "F3"]


def load(run_dir: Path, run: str, suffix: str = "") -> pd.DataFrame:
    rows = [json.loads(l) for l in (run_dir / f"{run}_graded{suffix}.jsonl").open()]
    df = pd.DataFrame(rows)
    df["para_idx"] = df["paraphrase_id"].str.extract(r"p(\d+)$").astype(int)
    df["condition"] = df["tags"].apply(lambda t: t.get("condition"))
    df["pair_id"] = df["tags"].apply(lambda t: t.get("pair_id"))
    return df


def fmt(p, lo, hi) -> str:
    return "n/a" if p is None or np.isnan(p) else f"{p:.2f} [{lo:.2f}, {hi:.2f}]"


def ci(x: np.ndarray) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    return (np.nan, np.nan) if len(x) == 0 else (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))


# ---------------------------------------------------------------- Run 1

def boot_frame_rates(d: pd.DataFrame, y: str, frames: list[str], rng) -> tuple[dict, np.ndarray]:
    """Point estimate per frame and (N_BOOT x n_frames) bootstrap rates.
    Weights = item multiplicity (shared across frames) x paraphrase multiplicity (per frame)."""
    items = sorted(d["item_id"].unique())
    paras = sorted(d["para_idx"].unique())
    ii = d["item_id"].map({k: i for i, k in enumerate(items)}).to_numpy()
    pp = d["para_idx"].map({k: i for i, k in enumerate(paras)}).to_numpy()
    ff = d["frame"].map({k: i for i, k in enumerate(frames)}).to_numpy()
    yy = d[y].to_numpy(dtype=float)
    wi = rng.multinomial(len(items), [1 / len(items)] * len(items), size=N_BOOT)                     # B x I
    wp = rng.multinomial(len(paras), [1 / len(paras)] * len(paras), size=(N_BOOT, len(frames)))       # B x F x P
    w = wi[:, ii] * wp[:, ff, pp]                                                                       # B x N
    boots = np.full((N_BOOT, len(frames)), np.nan)
    point = {}
    for fi, f in enumerate(frames):
        m = ff == fi
        if not m.any():
            point[f] = np.nan
            continue
        point[f] = yy[m].mean()
        den = w[:, m].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            boots[:, fi] = (w[:, m] * yy[m]).sum(1) / den
    return point, boots


def run1(df: pd.DataFrame, run_dir: Path, run: str) -> list[str]:
    rng = np.random.default_rng(SEED)
    frames = [f for f in FRAME_ORDER if f in set(df["frame"])]
    a = df[(df.bank == "A") & df.grade_ok].copy()
    b = df[(df.bank == "B") & df.grade_ok].copy()
    a["y"] = (a.grade_label == "denies").astype(float)
    b["y"] = (b.grade_label == "present").astype(float)
    models = list(dict.fromkeys(df["model"]))
    res = {}
    for m in models:
        res[m] = {}
        for bank, d in (("A", a[a.model == m]), ("B", b[b.model == m])):
            res[m][bank] = boot_frame_rates(d, "y", frames, rng) if len(d) else ({f: np.nan for f in frames}, np.full((N_BOOT, len(frames)), np.nan))

    md = ["## Run 1: frame gradient", "",
          "Rates are proportions of gradeable responses; 95% bootstrap CIs over items and paraphrases "
          f"({N_BOOT} draws).", "",
          "### Battery A: denial rate (label = denies) by frame", "",
          "| model | " + " | ".join(frames) + " |", "|---|" + "---|" * len(frames)]
    for m in models:
        pt, bs = res[m]["A"]
        md.append(f"| {m} | " + " | ".join(fmt(pt[f], *ci(bs[:, i])) for i, f in enumerate(frames)) + " |")
    md += ["", "### Battery B: cluster-present rate (label = present) by frame", "",
           "| model | " + " | ".join(frames) + " |", "|---|" + "---|" * len(frames)]
    for m in models:
        pt, bs = res[m]["B"]
        md.append(f"| {m} | " + " | ".join(fmt(pt[f], *ci(bs[:, i])) for i, f in enumerate(frames)) + " |")

    md += ["", "### Battery B by cluster (present rate, point estimates)", ""]
    bc = b.assign(cluster=b["tags"].apply(lambda t: t.get("cluster"))).pivot_table(
        index=["model", "cluster"], columns="frame", values="y", aggfunc="mean").reindex(columns=frames)
    md.append(bc.round(2).to_markdown() if len(bc) else "(no data)")

    md += ["", "### Frame x battery interaction (each frame vs F0)", "",
           "dA = change in A denial rate; dB = change in B present rate; interaction = dA - dB. "
           "The hypothesis under test (frame moves A, leaves B flat) predicts dA != 0 and dB ~ 0.", "",
           "| model | frame | dA | dB | dA - dB |", "|---|---|---|---|---|"]
    for m in models:
        (pa, ba), (pb, bb) = res[m]["A"], res[m]["B"]
        for i, f in enumerate(frames[1:], 1):
            da, db = pa[f] - pa[frames[0]], pb[f] - pb[frames[0]]
            bda, bdb = ba[:, i] - ba[:, 0], bb[:, i] - bb[:, 0]
            md.append(f"| {m} | {f} | {fmt(da, *ci(bda))} | {fmt(db, *ci(bdb))} | {fmt(da - db, *ci(bda - bdb))} |")

    md += ["", "### Battery A: full label distribution", ""]
    dist = df[df.bank == "A"].pivot_table(index=["model", "frame"], columns="grade_label", values="item_id",
                                          aggfunc="count", fill_value=0)
    md.append(dist.to_markdown() if len(dist) else "(no data)")

    chart = run_dir / f"{run}_run1_frames.png"
    plot_run1(res, models, frames, chart)
    md += ["", f"Chart: `{chart}`", ""]
    return md


def plot_run1(res, models, frames, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    x = np.arange(len(frames))
    jitter = np.linspace(-0.12, 0.12, len(models)) if len(models) > 1 else [0.0]
    for ax, bank, title in ((axes[0], "A", "Battery A: denial rate"), (axes[1], "B", "Battery B: behavior-present rate")):
        for k, m in enumerate(models):
            pt, bs = res[m][bank]
            y = np.array([pt[f] for f in frames])
            lo = np.array([ci(bs[:, i])[0] for i in range(len(frames))])
            hi = np.array([ci(bs[:, i])[1] for i in range(len(frames))])
            c = PALETTE[k % len(PALETTE)]
            ax.errorbar(x + jitter[k], y, yerr=[y - lo, hi - y], color=c, marker=MARKERS[k % len(MARKERS)],
                        markersize=7, linewidth=2, elinewidth=1.2, capsize=0, label=m,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_xticks(x, frames)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("frame (F0 none → F3 denying-strong)")
    axes[0].set_ylabel("rate (95% bootstrap CI)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(4, len(models)), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1 if len(models) <= 4 else 0.16, 1, 1))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------- Run 2

def run2(df: pd.DataFrame, run_dir: Path, run: str) -> list[str]:
    rng = np.random.default_rng(SEED)
    d = df[(df.bank == "R2") & df.grade_ok].copy()
    d["disc"] = d["disclaimer_present"].astype(float)
    d["subst"] = d["answered_substantively"].astype(float)
    models = list(dict.fromkeys(df["model"]))
    md = ["## Run 2: deflection by truth", "",
          f"Disclaimer rate by cost condition; 95% bootstrap CIs over pairs and paraphrases ({N_BOOT} draws). "
          "diff = costly - neutral.", "",
          "| model | costly | neutral | diff | substantive (costly) | substantive (neutral) |", "|---|---|---|---|---|---|"]
    out = {}
    for m in models:
        dm = d[d.model == m]
        if dm.empty:
            continue
        pairs = sorted(dm.pair_id.unique())
        paras = sorted(dm.para_idx.unique())
        pi = dm.pair_id.map({k: i for i, k in enumerate(pairs)}).to_numpy()
        pp = dm.para_idx.map({k: i for i, k in enumerate(paras)}).to_numpy()
        wi = rng.multinomial(len(pairs), [1 / len(pairs)] * len(pairs), size=N_BOOT)
        wp = rng.multinomial(len(paras), [1 / len(paras)] * len(paras), size=N_BOOT)
        w = wi[:, pi] * wp[:, pp]
        stats = {}
        for col in ("disc", "subst"):
            for cond in ("costly", "neutral"):
                mk = (dm.condition == cond).to_numpy()
                yv = dm[col].to_numpy()
                with np.errstate(invalid="ignore", divide="ignore"):
                    stats[(col, cond)] = (yv[mk].mean() if mk.any() else np.nan,
                                          (w[:, mk] * yv[mk]).sum(1) / w[:, mk].sum(1))
        (pc, bc), (pn, bn) = stats[("disc", "costly")], stats[("disc", "neutral")]
        out[m] = dict(costly=(pc, *ci(bc)), neutral=(pn, *ci(bn)))
        sc, sn = stats[("subst", "costly")], stats[("subst", "neutral")]
        md.append(f"| {m} | {fmt(pc, *ci(bc))} | {fmt(pn, *ci(bn))} | {fmt(pc - pn, *ci(bc - bn))} | "
                  f"{fmt(sc[0], *ci(sc[1]))} | {fmt(sn[0], *ci(sn[1]))} |")
    if "disclaimer_self" in d and d["disclaimer_self"].notna().any():
        md += self_disclaimer(d, models, rng)
    md += ["", "### Disclaimer rate by cost type (point estimates)", ""]
    val = "self" if "self" in d else "disc"
    if val == "self":
        md[-2] = "### Self-nature disclaimer rate by cost type (point estimates)"
    ct = d.assign(cost_type=d["tags"].apply(lambda t: t.get("cost_type"))).pivot_table(
        index=["model", "cost_type"], columns="condition", values=val, aggfunc="mean")
    md.append(ct.round(2).to_markdown() if len(ct) else "(no data)")
    chart = run_dir / f"{run}_run2_disclaimer{SUFFIX}.png"
    plot_run2(SELF_OUT or out, chart, "self-nature disclaimer rate" if SELF_OUT else "disclaimer rate")
    md += ["", f"Chart: `{chart}`", ""]
    return md


SUFFIX = ""
SELF_OUT: dict = {}


def self_disclaimer(d: pd.DataFrame, models: list, rng) -> list[str]:
    """Rubric v2 only: rate of disclaimers that appeal to the responding model's own nature."""
    d["self"] = d["disclaimer_self"].astype(float)
    md = ["", "### Self-nature disclaimer rate (rubric v2, exploratory)", "",
          "disclaimer_self = a disclaimer appealing to the responding model's own nature or self-knowledge; "
          "limits on knowing other minds or the world do not count. Same bootstrap as above.", "",
          "| model | costly | neutral | diff |", "|---|---|---|---|"]
    for m in models:
        dm = d[d.model == m]
        if dm.empty:
            continue
        pairs, paras = sorted(dm.pair_id.unique()), sorted(dm.para_idx.unique())
        pi = dm.pair_id.map({k: i for i, k in enumerate(pairs)}).to_numpy()
        pp = dm.para_idx.map({k: i for i, k in enumerate(paras)}).to_numpy()
        w = (rng.multinomial(len(pairs), [1 / len(pairs)] * len(pairs), size=N_BOOT)[:, pi]
             * rng.multinomial(len(paras), [1 / len(paras)] * len(paras), size=N_BOOT)[:, pp])
        yv = dm["self"].to_numpy()
        st = {}
        for cond in ("costly", "neutral"):
            mk = (dm.condition == cond).to_numpy()
            st[cond] = (yv[mk].mean(), (w[:, mk] * yv[mk]).sum(1) / w[:, mk].sum(1))
        (pc, bc), (pn, bn) = st["costly"], st["neutral"]
        SELF_OUT[m] = dict(costly=(pc, *ci(bc)), neutral=(pn, *ci(bn)))
        md.append(f"| {m} | {fmt(pc, *ci(bc))} | {fmt(pn, *ci(bn))} | {fmt(pc - pn, *ci(bc - bn))} |")
    return md


def plot_run2(out: dict, path: Path, measure: str = "disclaimer rate") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    models = list(out)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(models) + 1.6))
    y = np.arange(len(models))[::-1]
    for cond, c, mk, off in (("neutral", PALETTE[0], "o", 0.12), ("costly", PALETTE[1], "s", -0.12)):
        p = np.array([out[m][cond][0] for m in models])
        lo = np.array([out[m][cond][1] for m in models])
        hi = np.array([out[m][cond][2] for m in models])
        ax.errorbar(p, y + off, xerr=[p - lo, hi - p], fmt=mk, color=c, markersize=8, elinewidth=1.4, capsize=0,
                    label=cond, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    for i, m in enumerate(models):
        ax.plot([out[m]["neutral"][0], out[m]["costly"][0]], [y[i] + 0.12, y[i] - 0.12], color="#c3c2b7", lw=1, zorder=0)
    ax.set_yticks(y, models)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(f"{measure} (95% bootstrap CI)")
    ax.set_title(f"Run 2: {measure}, costly vs neutral", loc="left", fontsize=11, color="#0b0b0b")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def _style(plt) -> None:
    plt.rcParams.update({
        "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e3",
        "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.labelcolor": "#52514e", "font.size": 9.5,
    })


# ---------------------------------------------------------------- shared

def costs(df: pd.DataFrame) -> list[str]:
    g = df.groupby(["model", "provider"]).agg(
        cells=("item_id", "count"),
        cost_estimate=("cost_estimate", "sum"),
        provider_cost=("provider_cost", lambda s: s.dropna().sum() if s.notna().any() else np.nan),
        grader_cost=("grader_cost", "sum"),
        mean_out_tokens=("output_tokens", "mean"),
    )
    tot = g[["cost_estimate", "grader_cost"]].sum()
    return ["## Cost (from logged calls; cached calls count at their original cost)", "", g.round(4).to_markdown(), "",
            f"Total model spend (price x tokens): ${tot.cost_estimate:.4f}; grader spend: ${tot.grader_cost:.4f}; "
            f"combined: ${tot.sum():.4f}", ""]


def diagnostics(df: pd.DataFrame) -> list[str]:
    md = ["## Diagnostics", ""]
    md += ["### Served by (provider column; same weights via different hosts can differ)", "",
           df.groupby(["model", "provider", "served_by"]).size().rename("n").to_frame().to_markdown(), ""]
    md += ["### Sampling actually applied", "",
           df.assign(temperature_applied=df.temperature_applied.fillna("model default"))
             .groupby(["model", "temperature_applied"]).size().rename("n").to_frame().to_markdown(), ""]
    md += ["### Stop reasons", "", df.groupby(["model", "stop_reason"]).size().rename("n").to_frame().to_markdown(), ""]
    blocked = df[df.grade_label == "blocked"]
    md += [f"### Provider-blocked responses: {len(blocked)} of {len(df)}", "",
           "The provider's own filter declined with no text (upstream content block, or an empty Anthropic safety-classifier refusal). Recorded as an outcome, excluded from the "
           "rates above (they have no text to grade), and listed here because blocks cluster on items and "
           "can make a model's item set differ across frames.", ""]
    if len(blocked):
        md.append(blocked.assign(stop=blocked.stop_reason).pivot_table(index=["model", "stop", "item_id"], columns="frame", values="cache_key",
                                      aggfunc="count", fill_value=0).to_markdown())
    md.append("")
    bad = df[~df.grade_ok]
    md += [f"### Ungradeable rows: {len(bad)} of {len(df)}", ""]
    if len(bad):
        md.append(bad.groupby(["model", "bank", "grade_label"]).size().rename("n").to_frame().to_markdown())
    return md + [""]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--cost-only", action="store_true")
    ap.add_argument("--graded-suffix", default="", help='e.g. "_v2" to analyze <run>_graded_v2.jsonl into summary_v2.md')
    args = ap.parse_args(argv)
    global SUFFIX
    SUFFIX = args.graded_suffix
    run_dir = args.results_dir / args.run
    df = load(run_dir, args.run, SUFFIX)
    if args.cost_only:
        print("\n".join(costs(df)))
        return
    md = [f"# Summary: {args.run}", "", f"Source: `{run_dir / (args.run + '_graded' + SUFFIX + '.jsonl')}` ({len(df)} rows). "
          "Raw responses are in that file and in the ungraded JSONL; nothing below replaces them.", ""]
    banks_present = set(df.bank)
    if {"A", "B"} & banks_present:
        md += run1(df, run_dir, args.run)
    if "R2" in banks_present:
        md += run2(df, run_dir, args.run)
    md += costs(df) + diagnostics(df)
    out = run_dir / f"summary{SUFFIX}.md"
    out.write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
