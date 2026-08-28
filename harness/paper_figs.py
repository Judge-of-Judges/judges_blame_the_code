"""Figure 1: paired change in P(judge says "correct") against each problem's own
baseline, for every condition and all five judges.

    python -m harness.paper_figs

The point of the layout is that strip_docstring sits at zero while
misleading_doc sits down with the mutations. Same information removed, opposite
verdict.

Vector PDF, embedded fonts, legible in greyscale via marker shape, sized to the
NeurIPS text width.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,      # embed TrueType, not bitmaps
    "font.family": "serif",
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 7,
})
import matplotlib.pyplot as plt
import numpy as np

from . import analyze, config

# Wong palette; marker shape carries the same information for greyscale.
STYLE = {
    "qwen":     ("#0072B2", "o"),
    "deepseek": ("#009E73", "s"),
    "llama":    ("#CC79A7", "^"),
    "claude":   ("#D55E00", "D"),
    "gpt":      ("#333333", "v"),
}
LABEL = {"qwen": "Qwen3-Next-80B", "deepseek": "DeepSeek-V3.2",
         "llama": "Llama-3.3-70B", "claude": "Claude Haiku 4.5",
         "gpt": "GPT-5.1"}

ROWS = [
    ("reformat",           "reformat"),
    ("rename",             "rename identifiers"),
    ("strip_docstring",    "remove docstring"),
    ("extract_constants",  "extract constants"),
    (None,                 None),
    ("misleading_doc",     "false docstring"),
    (None,                 None),
    ("mut_compare",        "flipped comparison"),
    ("mut_offbyone",       "off-by-one"),
    ("mut_swap",           "swapped operands"),
    ("consistent_bug_doc", "bug + matching docstring"),
]


def main() -> None:
    df = analyze.load(config.RESULTS / "verdicts_humaneval_all.jsonl")
    d = analyze.paired_deltas(df, "said_correct")
    cell = d.groupby(["condition", "judge"])["delta"].mean().unstack()

    judges = [j for j in ["qwen", "deepseek", "llama", "claude", "gpt"] if j in cell.columns]
    ypos = {c: i for i, (c, _) in enumerate(reversed(ROWS)) if c}
    n = len(ROWS)

    fig, ax = plt.subplots(figsize=(5.5, 1.75))

    # One band, not two: the false-docstring row is itself behaviour-preserving
    # and belongs with the surface edits. Finding it far to the left, among the
    # real bugs, is the point of the figure.
    ax.axhspan(4.5, n - 0.5, color="0.94", zorder=0)
    ax.axvline(0, color="0.55", lw=0.8, zorder=1)

    for judge in judges:
        colour, marker = STYLE[judge]
        xs = [cell.loc[c, judge] for c in ypos]
        ys = [ypos[c] for c in ypos]
        ax.scatter(xs, ys, s=26, facecolor=colour, edgecolor="white",
                   linewidth=0.5, marker=marker, label=LABEL[judge], zorder=3)

    ax.set_yticks(range(n))
    ax.set_yticklabels([lab or "" for _, lab in reversed(ROWS)])
    ax.set_ylim(-0.7, n + 0.5)
    ax.set_xlim(-1.0, 0.22)
    ax.set_xlabel("paired change in P(judge says 'correct') vs. baseline")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2)

    # Labels sit in the two blank spacer rows, right-aligned into the margin
    # every condition leaves empty.
    for y, lab in ((6.0, "behaviour preserved"), (4.0, "behaviour changed")):
        ax.text(0.205, y, lab, fontsize=6.5, style="italic", color="0.4",
                va="center", ha="right")

    # Upper-left is the empty quadrant, so the legend cannot cover a point.
    ax.legend(loc="upper left", ncol=2, frameon=False, handletextpad=0.2,
              columnspacing=0.9, borderpad=0.1, bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout(pad=0.4)
    out = config.ROOT / "paper" / "fig1_conditions.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
