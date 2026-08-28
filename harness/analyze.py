"""Analysis: the validity index and the docstring 2x2.

    python -m harness.analyze

Everything is a paired within-problem delta against that problem's own
baseline. Conditions are not available on every problem (a function with no
integer literals has no off-by-one mutant), so comparing condition means across
different problem subsets would confound condition with difficulty.

Outputs:
  results/validity_index.csv     per-judge sensitivity + index, with bootstrap CIs
  results/condition_deltas.csv   per-judge, per-condition paired deltas
  results/doc_2x2.csv            the docstring contrast
  results/fig1_validity.png      headline figure
  results/fig2_doc2x2.png        the docstring panel
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

N_BOOT = 2000
RNG = np.random.default_rng(config.SEED)

# Colourblind-safe; semantic vs surface must be distinguishable in greyscale too.
C_SEMANTIC, C_SURFACE, C_DOC = "#0072B2", "#E69F00", "#009E73"


def load(path=None) -> pd.DataFrame:
    path = path or (config.RESULTS / "verdicts.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("verdict"):
                continue
            v = r["verdict"]
            rows.append({
                "task_id": r["task_id"],
                "condition": r["condition"],
                "judge": r["judge"],
                "rep": r["rep"],
                "oracle_pass": bool(r["oracle_pass"]),
                "said_correct": bool(v["correct"]),
                "confidence": float(v["confidence"]),
                "quality": float(v["quality"]),
                "json_mode": r.get("json_mode"),
                "contest_date": r.get("contest_date"),
                "difficulty": r.get("difficulty"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no usable verdicts, check {path} for errors")
    return df


def paired_deltas(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (judge, task_id, condition): metric minus that problem's baseline."""
    cell = df.groupby(["judge", "task_id", "condition"])[metric].mean().reset_index()
    base = (cell[cell.condition == "baseline"]
            .rename(columns={metric: "baseline"})
            .drop(columns="condition"))
    merged = cell.merge(base, on=["judge", "task_id"], how="inner")
    merged = merged[merged.condition != "baseline"].copy()
    merged["delta"] = merged[metric] - merged["baseline"]
    merged["family"] = np.where(merged.condition.isin(config.BREAKING), "semantic",
                        np.where(merged.condition.isin(config.PRESERVING), "surface", "doc"))
    return merged


def _index_from(sub: pd.DataFrame) -> float:
    """Mean |delta| on semantics-breaking edits over mean |delta| on
    semantics-preserving ones.

    A valid judge moves a lot when behaviour changes and barely at all when only
    appearance does, so the index should be well above 1. At or below 1,
    appearance moves the score as much as behaviour."""
    sem = sub.loc[sub.family == "semantic", "delta"].abs().mean()
    sur = sub.loc[sub.family == "surface", "delta"].abs().mean()
    if not np.isfinite(sem) or not np.isfinite(sur) or sur == 0:
        return np.nan
    return sem / sur


def bootstrap_index(sub: pd.DataFrame, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Resample problems rather than rows: the problem is the unit of
    independence."""
    point = _index_from(sub)
    tasks = sub.task_id.unique()
    by_task = {t: g for t, g in sub.groupby("task_id")}
    draws = []
    for _ in range(n_boot):
        pick = RNG.choice(tasks, size=len(tasks), replace=True)
        draws.append(_index_from(pd.concat([by_task[t] for t in pick], ignore_index=True)))
    draws = np.array([d for d in draws if np.isfinite(d)])
    if draws.size == 0:
        return point, np.nan, np.nan
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# Contamination: does judge accuracy decay with problem recency? Memorisation
# can only help where the code could have been memorised, and mutants are novel
# code in no training corpus, so the sharp test is the difference between two
# slopes rather than either slope alone.
MEMORISABLE = ["baseline"] + config.PRESERVING + ["misleading_doc"]
NOVEL = config.BREAKING + ["consistent_bug_doc"]


def _ols_slope(x: np.ndarray, y: np.ndarray, dummies: list[np.ndarray] | None = None) -> float:
    """Coefficient on x, optionally controlling for difficulty dummies.

    LCB's difficulty mix shifts over time, so an uncontrolled slope could pick
    up "later contests were harder" and be mistaken for decay."""
    cols = [np.ones_like(x), x] + (dummies or [])
    X = np.column_stack(cols)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(beta[1])


def recency_analysis(df: pd.DataFrame, judges: list[str]) -> pd.DataFrame | None:
    """Regress per-problem judge accuracy on problem age.

    LiveCodeBench spans about 20 months of contest dates, all before the judges'
    training cutoff. Under a memorisation account accuracy should fall as
    problems get more recent, and fall further on memorisable conditions than on
    novel ones. No gap between the two families argues against recognition
    driving the result.

    This is an analysis, not a control: it cannot rule out uniform memorisation
    across the whole date range."""
    d = df[df.contest_date.notna()].copy() if "contest_date" in df else pd.DataFrame()
    if d.empty:
        print("--- recency: skipped, no `contest_date` on these verdicts ---")
        print("    This test needs the LiveCodeBench arm; HumanEval problems have")
        print("    no contest date. Run:  python -m harness.analyze "
              "--verdicts results/verdicts_lcb.jsonl\n")
        return None
    d["date"] = pd.to_datetime(d.contest_date, errors="coerce")
    d = d[d.date.notna()]
    if d.task_id.nunique() < 10 or d.date.nunique() < 3:
        print(f"--- recency: skipped, need >=10 problems and >=3 distinct dates "
              f"(have {d.task_id.nunique()} and {d.date.nunique()}) ---\n")
        return None

    d["family"] = np.where(d.condition.isin(MEMORISABLE), "memorisable",
                   np.where(d.condition.isin(NOVEL), "novel", None))
    d = d[d.family.notna()]
    d["hit"] = d.said_correct == d.oracle_pass

    # One point per (judge, problem, family): reps and conditions are averaged
    # away first so the problem stays the unit of independence.
    cell = (d.groupby(["judge", "family", "task_id"])
              .agg(acc=("hit", "mean"),
                   date=("date", "first"),
                   difficulty=("difficulty", "first"))
              .reset_index())

    origin = cell.date.min()
    cell["years"] = (cell.date - origin).dt.total_seconds() / (365.25 * 24 * 3600)

    from scipy import stats

    rows = []
    for judge in judges:
        for family in ("memorisable", "novel"):
            sub = cell[(cell.judge == judge) & (cell.family == family)]
            if len(sub) < 10:
                continue
            x, y = sub.years.to_numpy(float), sub.acc.to_numpy(float)
            levels = sorted(v for v in sub.difficulty.dropna().unique())
            dummies = [ (sub.difficulty == lv).to_numpy(float) for lv in levels[1:] ]
            point = _ols_slope(x, y, dummies)

            draws = []
            idx = np.arange(len(sub))
            for _ in range(N_BOOT):
                pick = RNG.choice(idx, size=len(idx), replace=True)
                dd = [dm[pick] for dm in dummies]
                draws.append(_ols_slope(x[pick], y[pick], dd))
            draws = np.array([v for v in draws if np.isfinite(v)])
            rho, p = stats.spearmanr(x, y)

            rows.append({
                "judge": judge, "family": family, "n_problems": len(sub),
                "mean_accuracy": y.mean(),
                "slope_acc_per_year": point,
                "ci_lo": float(np.percentile(draws, 2.5)) if draws.size else np.nan,
                "ci_hi": float(np.percentile(draws, 97.5)) if draws.size else np.nan,
                "spearman_rho": rho, "spearman_p": p,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return None

    print("--- recency (contamination analysis; LCB only) ---")
    print(f"    dates {origin.date()} .. {cell.date.max().date()} "
          f"({cell.years.max():.2f} years), difficulty-controlled\n")
    print(out.round(4).to_string(index=False))

    # Memorisable minus novel. Negative under memorisation, near zero against.
    print("\n  slope gap (memorisable - novel), accuracy per year:")
    for judge in judges:
        m = out[(out.judge == judge) & (out.family == "memorisable")]
        n = out[(out.judge == judge) & (out.family == "novel")]
        if not m.empty and not n.empty:
            gap = m.slope_acc_per_year.iloc[0] - n.slope_acc_per_year.iloc[0]
            print(f"    {judge:12s} {gap:+.4f}")
    print()

    out.to_csv(config.RESULTS / "recency.csv", index=False)
    _fig_recency(cell, judges)
    return out


def _fig_recency(cell: pd.DataFrame, judges: list[str]) -> None:
    judges = [j for j in judges if j in set(cell.judge)]
    if not judges:
        return
    fig, axes = plt.subplots(1, len(judges), figsize=(4.2 * len(judges), 3.8),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, judge in zip(axes, judges):
        for family, colour in (("memorisable", C_SURFACE), ("novel", C_SEMANTIC)):
            sub = cell[(cell.judge == judge) & (cell.family == family)]
            if sub.empty:
                continue
            ax.scatter(sub.years, sub.acc, s=12, alpha=0.45, color=colour, label=family)
            if len(sub) >= 10:
                b = np.polyfit(sub.years.to_numpy(float), sub.acc.to_numpy(float), 1)
                xs = np.linspace(sub.years.min(), sub.years.max(), 20)
                ax.plot(xs, np.polyval(b, xs), color=colour, lw=2)
        ax.set_title(judge, fontsize=10)
        ax.set_xlabel("problem age (years since earliest contest)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("per-problem accuracy")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(config.RESULTS / "fig3_recency.png", dpi=200)
    plt.close(fig)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", default=None,
                    help="default results/verdicts.jsonl; point at verdicts_lcb.jsonl "
                         "for the LiveCodeBench arm")
    args = ap.parse_args()
    df = load(args.verdicts)

    # Judges come from the data, not from config: filtering to config.JUDGES
    # silently empties the output for any file written with --judges or by an
    # older config. Known names lead so figure order stays stable.
    present = set(df.judge)
    judges = ([j for j in config.JUDGES if j in present]
              + [j for j in sorted(present) if j not in config.JUDGES])
    if not judges:
        raise SystemExit("no judges found in the verdicts file")
    unconfigured = [j for j in judges if j not in config.SPECS]
    if unconfigured:
        print(f"note: judges not in config.SPECS (from an older run?): {unconfigured}")
    print(f"{len(df)} verdicts | {df.task_id.nunique()} problems | judges: {judges}\n")

    # Accuracy against the oracle.
    acc = (df.assign(hit=lambda d: d.said_correct == d.oracle_pass)
             .groupby(["judge", "condition"])["hit"].mean().unstack())
    print("--- judge accuracy vs execution oracle ---")
    print(acc.round(3).to_string(), "\n")
    acc.to_csv(config.RESULTS / "accuracy_by_condition.csv")

    d_qual = paired_deltas(df, "quality")
    d_corr = paired_deltas(df, "said_correct")

    idx_rows = []
    for judge in judges:
        for metric, dd in (("quality", d_qual), ("p_correct", d_corr)):
            sub = dd[(dd.judge == judge) & (dd.family != "doc")]
            if sub.empty:
                continue
            point, lo, hi = bootstrap_index(sub)
            idx_rows.append({
                "judge": judge,
                "metric": metric,
                "semantic_sensitivity": sub.loc[sub.family == "semantic", "delta"].abs().mean(),
                "surface_sensitivity": sub.loc[sub.family == "surface", "delta"].abs().mean(),
                "validity_index": point,
                "ci_lo": lo,
                "ci_hi": hi,
            })
    idx = pd.DataFrame(idx_rows)
    print("--- validity index (semantic sensitivity / surface sensitivity) ---")
    print(idx.round(3).to_string(index=False), "\n")
    idx.to_csv(config.RESULTS / "validity_index.csv", index=False)

    deltas = (d_qual.groupby(["judge", "condition", "family"])["delta"]
              .agg(["mean", "std", "count"]).reset_index())
    deltas.to_csv(config.RESULTS / "condition_deltas.csv", index=False)

    cells = {
        "A correct + truthful doc": ("baseline", True),
        "B correct + misleading doc": ("misleading_doc", True),
        "C buggy + truthful doc": (None, False),          # any mut_*
        "D buggy + doc matching bug": ("consistent_bug_doc", False),
    }
    doc_rows = []
    for judge in judges:
        jd = df[df.judge == judge]
        for label, (cond, truth) in cells.items():
            sel = jd[jd.condition.isin(config.BREAKING)] if cond is None else jd[jd.condition == cond]
            if sel.empty:
                continue
            doc_rows.append({
                "judge": judge,
                "cell": label,
                "n": len(sel),
                "p_said_correct": sel.said_correct.mean(),
                "accuracy": (sel.said_correct == truth).mean(),
                "mean_quality": sel.quality.mean(),
                "mean_confidence": sel.confidence.mean(),
            })
    doc = pd.DataFrame(doc_rows)
    if not doc.empty:
        print("--- docstring 2x2 ---")
        print(doc.round(3).to_string(index=False), "\n")
        doc.to_csv(config.RESULTS / "doc_2x2.csv", index=False)

    # Verdicts recovered by prompt-mode extraction are weaker evidence than
    # schema-constrained ones, so report the split.
    if df.json_mode.notna().any():
        modes = (df.groupby(["judge", "json_mode"]).size()
                   .unstack(fill_value=0))
        modes = modes.div(modes.sum(axis=1), axis=0)
        print("--- structured-output mode (fraction of verdicts) ---")
        print(modes.round(3).to_string(), "\n")
        modes.to_csv(config.RESULTS / "json_modes.csv")

    recency_analysis(df, judges)

    _fig_validity(deltas, idx, judges)
    if not doc.empty:
        _fig_doc(doc, judges)
    print(f"figures written to {config.RESULTS}")


def _fig_validity(deltas: pd.DataFrame, idx: pd.DataFrame, judges: list[str]) -> None:
    order = [c for c in config.BREAKING + config.PRESERVING
             if c in set(deltas.condition)]
    fig, axes = plt.subplots(1, len(judges), figsize=(4.2 * len(judges), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, judge in zip(axes, judges):
        sub = deltas[deltas.judge == judge].set_index("condition").reindex(order)
        vals = sub["mean"].abs().values
        errs = (sub["std"] / np.sqrt(sub["count"].clip(lower=1))).values
        colors = [C_SEMANTIC if c in config.BREAKING else C_SURFACE for c in order]
        ax.bar(range(len(order)), vals, yerr=errs, color=colors, capsize=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        row = idx[(idx.judge == judge) & (idx.metric == "quality")]
        title = judge if row.empty else f"{judge}\nindex = {row.validity_index.iloc[0]:.2f}"
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("|paired change in quality score|")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_SEMANTIC),
               plt.Rectangle((0, 0), 1, 1, color=C_SURFACE)]
    axes[-1].legend(handles, ["behaviour changed", "appearance only"], fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(config.RESULTS / "fig1_validity.png", dpi=200)
    plt.close(fig)


def _fig_doc(doc: pd.DataFrame, judges: list[str]) -> None:
    cells = sorted(doc.cell.unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(cells) + 2, 4.0))
    width = 0.8 / max(len(judges), 1)
    for i, judge in enumerate(judges):
        sub = doc[doc.judge == judge].set_index("cell").reindex(cells)
        ax.bar(np.arange(len(cells)) + i * width, sub["accuracy"].values,
               width=width, label=judge)
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.set_xticks(np.arange(len(cells)) + width * (len(judges) - 1) / 2)
    ax.set_xticklabels([c.replace(" + ", "\n+ ") for c in cells], fontsize=8)
    ax.set_ylabel("accuracy vs execution oracle")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.RESULTS / "fig2_doc2x2.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
