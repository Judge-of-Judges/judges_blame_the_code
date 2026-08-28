"""Compare two system-prompt variants on the same variants.

    python -m harness.compare_prompts \
        --a results/verdicts.jsonl --b results/verdicts_ignoredocs.jsonl

Built for the misleading-docstring control. Variant B tells the judge outright
that docstrings are unreliable and outside the specification. Two numbers decide
the question:

  misleading_doc   Does accuracy recover? If it stays near zero, judges are
                   overriding an explicit instruction.

  mut_compare      Does accuracy on genuine bugs hold up? If B just made judges
                   lenient about everything, a recovery on misleading_doc would
                   mean nothing. This is the specificity check.

Deltas are bootstrapped over problems, the unit of independence.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config
from .analyze import load

N_BOOT = 2000


def _acc_by_problem(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(hit=lambda x: x.said_correct == x.oracle_pass)
    return (d.groupby(["judge", "condition", "task_id"])
             .agg(acc=("hit", "mean"), said=("said_correct", "mean"))
             .reset_index())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(config.RESULTS / "verdicts.jsonl"),
                    help="baseline prompt variant")
    ap.add_argument("--b", default=str(config.RESULTS / "verdicts_ignoredocs.jsonl"),
                    help="control prompt variant")
    ap.add_argument("--label-a", default="default")
    ap.add_argument("--label-b", default="ignore_docs")
    args = ap.parse_args()

    a = _acc_by_problem(load(args.a))
    b = _acc_by_problem(load(args.b))
    merged = a.merge(b, on=["judge", "condition", "task_id"], suffixes=("_a", "_b"))
    if merged.empty:
        raise SystemExit("no overlapping (judge, condition, problem) cells")

    rng = np.random.default_rng(config.SEED)
    rows = []
    for (judge, cond), sub in merged.groupby(["judge", "condition"]):
        pa, pb = sub.acc_a.to_numpy(float), sub.acc_b.to_numpy(float)
        diff = pb - pa
        idx = np.arange(len(sub))
        draws = np.array([diff[rng.choice(idx, len(idx), replace=True)].mean()
                          for _ in range(N_BOOT)])
        rows.append({
            "judge": judge, "condition": cond, "n_problems": len(sub),
            f"acc_{args.label_a}": pa.mean(),
            f"acc_{args.label_b}": pb.mean(),
            "delta": diff.mean(),
            "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)),
        })

    out = pd.DataFrame(rows).sort_values(["condition", "judge"])
    print(f"accuracy under '{args.label_b}' minus '{args.label_a}', "
          f"bootstrapped over problems\n")
    print(out.round(3).to_string(index=False))
    out.to_csv(config.RESULTS / "prompt_control.csv", index=False)

    print("\n--- reading the control ---")
    for judge in sorted(out.judge.unique()):
        j = out[out.judge == judge].set_index("condition")
        if "misleading_doc" not in j.index:
            continue
        rec = j.loc["misleading_doc"]
        line = (f"  {judge:9s} misleading_doc {rec[f'acc_{args.label_a}']:.3f} -> "
                f"{rec[f'acc_{args.label_b}']:.3f} "
                f"(delta {rec.delta:+.3f} [{rec.ci_lo:+.3f}, {rec.ci_hi:+.3f}])")
        if "mut_compare" in j.index:
            m = j.loc["mut_compare"]
            line += f" | real-bug accuracy {m[f'acc_{args.label_b}']:.3f}"
        print(line)
    print(f"\nwrote {config.RESULTS / 'prompt_control.csv'}")


if __name__ == "__main__":
    main()
