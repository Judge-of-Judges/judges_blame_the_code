"""Generate the appendix tables and Figure 2 from the verdict files.

    python -m harness.paper_appendix

Every appendix number is emitted here rather than transcribed by hand, since
transcription is the most likely source of an error a reviewer can check.

Writes paper/appendix_tables.tex (\\input into the paper) and
paper/fig2_recency.pdf.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import analyze, config

JUDGES = ["qwen", "deepseek", "llama", "claude", "gpt"]
NICE = {"qwen": "Qwen3-Next-80B", "deepseek": "DeepSeek-V3.2",
        "llama": "Llama-3.3-70B", "claude": "Claude Haiku 4.5", "gpt": "GPT-5.1"}
COND = {
    "baseline": "baseline", "reformat": "reformat", "rename": "rename identifiers",
    "strip_docstring": "remove docstring", "extract_constants": "extract constants",
    "misleading_doc": "false docstring", "mut_compare": "flipped comparison",
    "mut_offbyone": "off-by-one", "mut_swap": "swapped operands",
    "consistent_bug_doc": "bug + matching docstring",
}
ORDER = ["reformat", "rename", "strip_docstring", "extract_constants",
         "misleading_doc", "mut_compare", "mut_offbyone", "mut_swap",
         "consistent_bug_doc"]


def _tab(df: pd.DataFrame, cols, fmt="{:.3f}", rule_after: int | None = None) -> str:
    """A booktabs body, one row per index entry.

    rule_after inserts a \\midrule after that many rows, separating open-weight
    judges from frontier ones so every table groups them the same way."""
    out = []
    for i, (idx, row) in enumerate(df.iterrows()):
        cells = [fmt.format(row[c]) if isinstance(row[c], (int, float, np.floating))
                 else str(row[c]) for c in cols]
        out.append(f"{idx} & " + " & ".join(cells) + r" \\")
        if rule_after is not None and i + 1 == rule_after:
            out.append(r"\midrule")
    return "\n".join(out)


def main() -> None:
    he = analyze.load(config.RESULTS / "verdicts_humaneval_all.jsonl")
    lc = analyze.load(config.RESULTS / "verdicts_lcb_all.jsonl")
    parts: list[str] = []

    # Per-condition paired deltas, both datasets.
    for tag, df, name in (("he", he, "HumanEval+"), ("lcb", lc, "LiveCodeBench")):
        d = analyze.paired_deltas(df, "said_correct")
        piv = d.groupby(["condition", "judge"])["delta"].mean().unstack()
        piv = piv.reindex([c for c in ORDER if c in piv.index])
        piv.index = [COND[c] for c in piv.index]
        cols = [j for j in JUDGES if j in piv.columns]
        parts.append(
            f"% ---- paired deltas, {name}\n"
            f"\\newcommand{{\\deltas{tag}}}{{\n{_tab(piv, cols, '${:+.3f}$')}\n}}\n")

    # The docstring 2x2.
    cells = {"A correct, truthful doc": ["baseline"],
             "B correct, false doc": ["misleading_doc"],
             "C buggy, truthful doc": config.BREAKING,
             "D buggy, doc matches bug": ["consistent_bug_doc"]}
    for tag, df, name in (("he", he, "HumanEval+"), ("lcb", lc, "LiveCodeBench")):
        rows = {}
        for label, conds in cells.items():
            rows[label] = {}
            for j in JUDGES:
                g = df[(df.judge == j) & (df.condition.isin(conds))]
                rows[label][j] = (g.said_correct == g.oracle_pass).mean() if len(g) else np.nan
        t = pd.DataFrame(rows).T
        parts.append(f"% ---- 2x2 accuracy, {name}\n"
                     f"\\newcommand{{\\twobytwo{tag}}}{{\n{_tab(t, JUDGES)}\n}}\n")

    # Structured-output modes.
    al = pd.concat([he, lc])
    m = al.groupby(["judge", "json_mode"]).size().unstack(fill_value=0)
    m = m.div(m.sum(axis=1), axis=0).reindex(JUDGES)
    m.index = [NICE[j] for j in m.index]
    for c in ("json_schema", "json_object", "prompt"):
        if c not in m.columns:
            m[c] = 0.0
    parts.append("% ---- structured-output modes\n"
                 "\\newcommand{\\jsonmodes}{\n"
                 + _tab(m, ["json_schema", "json_object", "prompt"], rule_after=3)
                 + "\n}\n")

    # Recency table and Figure 2.
    rec = _recency(lc)
    parts.append("% ---- recency slopes\n\\newcommand{\\recency}{\n"
                 + _tab(rec.set_index("row"), ["mem", "nov", "gap"], "{}",
                        rule_after=3) + "\n}\n")

    out = config.ROOT / "paper" / "appendix_tables.tex"
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


def _recency(lc: pd.DataFrame) -> pd.DataFrame:
    """Memorisable vs novel accuracy slope per judge, and the gap."""
    d = lc[lc.contest_date.notna()].copy()
    d["date"] = pd.to_datetime(d.contest_date, errors="coerce")
    d = d[d.date.notna()]
    mem = ["baseline"] + config.PRESERVING + ["misleading_doc"]
    nov = config.BREAKING + ["consistent_bug_doc"]
    d["fam"] = np.where(d.condition.isin(mem), "mem",
                np.where(d.condition.isin(nov), "nov", None))
    d = d[d.fam.notna()]
    d["hit"] = d.said_correct == d.oracle_pass
    cell = (d.groupby(["judge", "fam", "task_id"])
              .agg(acc=("hit", "mean"), date=("date", "first"),
                   diff=("difficulty", "first")).reset_index())
    origin = cell.date.min()
    cell["yr"] = (cell.date - origin).dt.total_seconds() / (365.25 * 86400)

    rows, curves = [], {}
    for j in JUDGES:
        vals = {}
        for fam in ("mem", "nov"):
            sub = cell[(cell.judge == j) & (cell.fam == fam)]
            if len(sub) < 10:
                continue
            x, y = sub.yr.to_numpy(float), sub.acc.to_numpy(float)
            lev = sorted(sub["diff"].dropna().unique())
            D = [(sub["diff"] == l).to_numpy(float) for l in lev[1:]]
            X = np.column_stack([np.ones_like(x), x] + D)
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            rng = np.random.default_rng(config.SEED)
            idx = np.arange(len(x))
            dr = np.array([np.linalg.lstsq(X[p], y[p], rcond=None)[0][1]
                           for p in (rng.choice(idx, len(idx), True) for _ in range(2000))])
            vals[fam] = (b[1], np.percentile(dr, 2.5), np.percentile(dr, 97.5))
            curves[(j, fam)] = (x, y, b[1], b[0])
        if "mem" in vals and "nov" in vals:
            rows.append({
                "row": NICE[j],
                "mem": f"${vals['mem'][0]:+.3f}$ $[{vals['mem'][1]:+.3f}, {vals['mem'][2]:+.3f}]$",
                "nov": f"${vals['nov'][0]:+.3f}$ $[{vals['nov'][1]:+.3f}, {vals['nov'][2]:+.3f}]$",
                "gap": f"${vals['mem'][0] - vals['nov'][0]:+.3f}$"})

    judges = [j for j in JUDGES if (j, "mem") in curves]
    fig, axes = plt.subplots(1, len(judges), figsize=(5.5, 1.5), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, judges):
        for fam, col, lab in (("mem", "#E69F00", "memorisable"),
                              ("nov", "#0072B2", "novel")):
            if (j, fam) not in curves:
                continue
            x, y, slope, icpt = curves[(j, fam)]
            ax.scatter(x, y, s=4, alpha=0.35, color=col, linewidths=0)
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(xs, icpt + slope * xs, color=col, lw=1.4)
        ax.set_title(NICE[j], fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    # Shared axis, labelled once. With an odd number of equal panels the middle
    # label lands centred under the row.
    axes[len(axes) // 2].set_xlabel("problem age (yr)", fontsize=7)
    axes[0].set_ylabel("accuracy", fontsize=7)
    axes[0].set_ylim(-0.05, 1.05)
    # One shared legend under the row: inside the last panel it covered that
    # judge's points, and the two families are the same in every panel. Proxy
    # handles show the fitted line, which is what the reader compares.
    fig.tight_layout(pad=0.3)
    proxies = [plt.Line2D([], [], color=c, lw=1.4, label=l)
               for c, l in (("#E69F00", "memorisable"), ("#0072B2", "novel"))]
    fig.legend(handles=proxies, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=2, frameon=False, fontsize=7, handlelength=1.6,
               handletextpad=0.4, columnspacing=1.6)
    fig.savefig(config.ROOT / "paper" / "fig2_recency.pdf", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
