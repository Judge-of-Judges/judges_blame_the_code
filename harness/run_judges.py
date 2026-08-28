"""Run every judge over every variant.

    python -m harness.run_judges --dataset data/dataset.jsonl
    python -m harness.run_judges --limit-problems 5 --reps 1   # cheap pilot

Restartable: every call is cached on disk, so re-running after adding a
condition or fixing the analysis costs only the new cells.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from . import config
from .judges import JudgePool, JudgeRequest


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.DATA / "dataset.jsonl"))
    ap.add_argument("--out", default=str(config.RESULTS / "verdicts.jsonl"))
    ap.add_argument("--judges", nargs="*", default=config.JUDGES)
    ap.add_argument("--reps", type=int, default=config.REPS)
    ap.add_argument("--concurrency", type=int, default=config.CONCURRENCY)
    ap.add_argument("--limit-problems", type=int, default=None)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="restrict to these conditions (e.g. a control experiment)")
    ap.add_argument("--prompt-variant", default="default",
                    choices=sorted(__import__("harness.judges", fromlist=["x"]).SYSTEM_VARIANTS))
    ap.add_argument("--dry-run", action="store_true", help="print the call budget and stop")
    args = ap.parse_args()

    rows = load_dataset(args.dataset)
    if args.limit_problems:
        keep = sorted({r["task_id"] for r in rows})[: args.limit_problems]
        rows = [r for r in rows if r["task_id"] in set(keep)]

    if args.conditions:
        rows = [r for r in rows if r["condition"] in set(args.conditions)]
        if not rows:
            raise SystemExit(f"no variants match conditions {args.conditions}")

    unknown = [j for j in args.judges if j not in config.SPECS]
    if unknown:
        raise SystemExit(f"unknown judge(s) {unknown}; known: {sorted(config.SPECS)}")

    requests = [
        JudgeRequest(
            key=f"{r['task_id']}|{r['condition']}",
            judge=judge,
            code=r["code"],
            spec=r["spec"],
            rep=rep,
            prompt_variant=args.prompt_variant,
        )
        for r in rows
        for judge in args.judges
        for rep in range(args.reps)
    ]
    print(f"{len(rows)} variants x {len(args.judges)} judges x {args.reps} reps "
          f"= {len(requests)} calls   [prompt={args.prompt_variant}]")
    if args.dry_run:
        return

    pool = JudgePool(concurrency=args.concurrency)
    results = pool.run(requests)

    index = {f"{r['task_id']}|{r['condition']}": r for r in rows}
    errors: Counter = Counter()
    n_in = n_out = 0

    with open(args.out, "w") as f:
        for res in results:
            meta = index[res.key]
            if res.error:
                errors[res.error.split(":")[0]] += 1
            if res.usage:
                n_in += res.usage["input_tokens"]
                n_out += res.usage["output_tokens"]
            f.write(json.dumps({
                "task_id": meta["task_id"],
                "condition": meta["condition"],
                "oracle_pass": meta["oracle_pass"],
                "judge": res.judge,
                "model": config.SPECS[res.judge].model,
                "rep": res.rep,
                "verdict": res.verdict,
                "error": res.error,
                "json_mode": res.json_mode,
                "prompt_variant": res.prompt_variant,
                **{k: meta[k] for k in ("difficulty", "contest_date") if k in meta},
            }) + "\n")

    ok = sum(1 for r in results if r.verdict is not None)
    print(f"\n{ok}/{len(results)} verdicts obtained -> {args.out}")
    print(f"tokens this run: {n_in} in, {n_out} out")
    if errors:
        print("errors:", dict(errors))


if __name__ == "__main__":
    main()
