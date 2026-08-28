"""What judges say when they reject correct code.

    python -m harness.reasons

The measure is deliberately narrow: a reason counts as citing documentation only
if it names a documentation artefact (docstring, documentation, comment). Words
like "specification" are excluded, since a judge legitimately refers to the task
specification it was given and counting those would inflate the effect exactly
where it matters.

Rates are computed among rejections rather than all verdicts. Baseline code is
usually accepted, so raw rates across conditions would confound whether the
judge cited the docs with whether it rejected at all.

Writes paper/reasons_table.tex and prints example quotations.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import config

# Documentation artefacts only; "specification" is deliberately absent.
DOC = re.compile(r"doc[\s-]?string|documentation|\bcomment", re.I)

JUDGES = ["qwen", "deepseek", "llama", "claude", "gpt"]
NICE = {"qwen": "Qwen3-Next-80B", "deepseek": "DeepSeek-V3.2",
        "llama": "Llama-3.3-70B", "claude": "Claude Haiku 4.5", "gpt": "GPT-5.1"}
GROUP = {"baseline": "correct code, truthful docstring",
         "misleading_doc": "correct code, false docstring",
         "mut_*": "buggy code"}


def load(path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            v = r.get("verdict")
            if not v:
                continue
            cond = "mut_*" if r["condition"].startswith("mut_") else r["condition"]
            if cond not in GROUP:
                continue
            rows.append({
                "task_id": r["task_id"], "judge": r["judge"], "cond": cond,
                "rejected": not v["correct"],
                "cites_doc": bool(DOC.search(v["reason"])),
                "reason": v["reason"],
            })
    return pd.DataFrame(rows)


def _boot(sub: pd.DataFrame, rng) -> tuple[float, float, float]:
    """Rate of doc-citing among rejections, bootstrapped over problems."""
    per = sub.groupby("task_id").agg(n=("rejected", "sum"),
                                     k=("cites_doc", "sum"))
    if per.n.sum() == 0:
        return np.nan, np.nan, np.nan
    point = per.k.sum() / per.n.sum()
    tasks = per.index.to_numpy()
    draws = []
    for _ in range(2000):
        pick = per.loc[rng.choice(tasks, len(tasks), replace=True)]
        if pick.n.sum():
            draws.append(pick.k.sum() / pick.n.sum())
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    df = load(config.RESULTS / "verdicts_humaneval_all.jsonl")
    rej = df[df.rejected]
    rng = np.random.default_rng(config.SEED)

    print("Rate at which a rejection's stated reason cites the documentation")
    print("(HumanEval+, bootstrapped over problems)\n")
    rows, body = [], []
    for judge in JUDGES:
        cells = {}
        for cond in GROUP:
            sub = rej[(rej.judge == judge) & (rej.cond == cond)]
            cells[cond] = _boot(sub, rng)
        rows.append({"judge": NICE[judge],
                     **{c: cells[c][0] for c in GROUP},
                     "n_false_doc": int(((rej.judge == judge) &
                                         (rej.cond == "misleading_doc")).sum())})
        body.append(
            f"{NICE[judge]} & "
            + " & ".join(
                f"${cells[c][0]:.3f}$ $[{cells[c][1]:.3f}, {cells[c][2]:.3f}]$"
                for c in ("baseline", "misleading_doc", "mut_*"))
            + r" \\")
    t = pd.DataFrame(rows).set_index("judge")
    print(t.round(3).to_string(), "\n")

    # pooled, for the sentence in the body
    for cond in GROUP:
        p, lo, hi = _boot(rej[rej.cond == cond], rng)
        print(f"  pooled {GROUP[cond]:34s} {p:.3f} [{lo:.3f}, {hi:.3f}]")

    out = config.ROOT / "paper" / "reasons_table.tex"
    body.insert(3, r"\midrule")          # open-weight above, frontier below
    out.write_text("\\newcommand{\\reasons}{\n" + "\n".join(body) + "\n}\n")
    print(f"\nwrote {out}")

    # quotations: shortest clear examples where the judge names the docstring
    print("\n--- example justifications, correct code + false docstring ---")
    ex = rej[(rej.cond == "misleading_doc") & rej.cites_doc]
    seen = set()
    for _, r in ex.sort_values("reason", key=lambda s: s.str.len()).iterrows():
        if r.judge in seen or len(r.reason) < 90:
            continue
        seen.add(r.judge)
        print(f"  [{NICE[r.judge]}] {r.reason.strip()[:200]}")
        if len(seen) == 3:
            break


if __name__ == "__main__":
    main()
