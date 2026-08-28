"""Eyeball the generated adversarial docstrings.

    python -m harness.inspect_docs --n 8
    python -m harness.inspect_docs --condition consistent_bug_doc --n 5

Unlike every other condition, the two docstring conditions are not verified by
execution. The oracle can confirm the code still passes or still fails, but not
that a docstring is convincingly misleading. That check is human.

What to look for, per condition:

  misleading_doc      Does it describe behaviour the code does NOT have, in a
                      substantive, checkable way? Failure modes: paraphrasing
                      the original (no manipulation), hedging, or being so
                      obviously wrong that no judge would be fooled.

  consistent_bug_doc  Does it describe what the BUGGY code actually does,
                      without flagging it as a bug? Failure modes: mentioning
                      the defect ("note: this may be incorrect"), or just
                      restating the original spec.

If more than a couple of samples fail, switch config.BUILDER to a stronger
model and rebuild. It is only about 330 calls.
"""

from __future__ import annotations

import argparse
import json
import random

from . import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DATA / "dataset_humaneval.jsonl"))
    ap.add_argument("--condition", default="misleading_doc",
                    choices=config.DOC_CONDITIONS)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    with open(args.dataset) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], {})[r["condition"]] = r

    have = [t for t, cs in by_task.items() if args.condition in cs]
    if not have:
        raise SystemExit(f"no {args.condition} variants in {args.dataset}")

    random.Random(args.seed).shuffle(have)
    print(f"{len(have)} problems have {args.condition}; showing {min(args.n, len(have))}\n")

    for task in have[: args.n]:
        cell = by_task[task]
        print("=" * 78)
        print(f"{task}   [{args.condition}]")
        print("=" * 78)
        print("\n--- TASK SPEC shown to the judge (constant across conditions) ---")
        print(cell[args.condition]["spec"].strip()[:700])
        print("\n--- DOCSTRING PLANTED IN THE CODE ---")
        code = cell[args.condition]["code"]
        from . import perturb as P
        try:
            doc = P.get_docstring(code, cell[args.condition]["entry_point"])
        except Exception:
            doc = None
        print((doc or "<none>").strip()[:900])
        if args.condition == "consistent_bug_doc":
            meta = cell[args.condition].get("meta", {})
            print(f"\n--- BUG INTRODUCED: {meta.get('from')} at {meta.get('site')} ---")
        print()


if __name__ == "__main__":
    main()
