"""Build the variant dataset, oracle-verified.

    python -m harness.build_dataset --limit 20            # pilot, no API needed
    python -m harness.build_dataset --skip-doc            # all 8 code conditions
    python -m harness.build_dataset                       # full, needs credentials

Every variant in the output file has been executed and agrees with its
condition's declared ORACLE_TRUTH. Anything that does not is dropped and
counted in the availability report rather than silently kept.

Output: data/dataset.jsonl, one record per (problem, condition).
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, oracle
from . import perturb as P

# Which mutant carries consistent_bug_doc, in order of preference. Comparison
# flips are cleanest since the bug is one character wide.
MUTANT_PREFERENCE = ["mut_compare", "mut_offbyone", "mut_swap"]


def build_problem(prob: dict, docgen=None) -> tuple[list[dict], Counter]:
    """All verified variants for one problem, plus a per-condition tally."""
    tally: Counter = Counter()
    entry = prob["entry_point"]
    out: list[dict] = []

    baseline = P.normalize(prob["canonical"])
    spec = P.get_docstring(baseline, entry)
    if not spec or not spec.strip():
        tally["skip_no_spec"] += 1
        return [], tally

    try:
        # The reference gets a longer budget than candidates: it runs once and
        # defines ground truth, so a slow canonical solution should not cost
        # the whole problem.
        ref = oracle.reference_outputs(baseline, entry, prob["inputs"],
                                       timeout=config.EXEC_TIMEOUT_S * 5)
    except oracle.OracleError as e:
        kind = "timeout" if "timeout" in str(e) else "crash"
        tally[f"skip_reference_{kind}"] += 1
        return [], tally

    if not oracle.passes(baseline, entry, prob["inputs"], ref):
        tally["skip_baseline_unstable"] += 1     # nondeterministic reference
        return [], tally

    # Problem-level metadata carried through to every variant. contest_date is
    # what the recency analysis regresses judge accuracy against.
    extras = {k: prob[k] for k in ("difficulty", "contest_date") if k in prob}

    def record(condition: str, code: str, meta: dict | None = None):
        out.append({
            "task_id": prob["task_id"],
            "entry_point": entry,
            "condition": condition,
            "code": code,
            "spec": spec,
            "oracle_pass": config.ORACLE_TRUTH[condition],
            "meta": meta or {},
            **extras,
        })
        tally[condition] += 1

    record("baseline", baseline)

    # Semantics-preserving: must still pass.
    for name, fn in P.PRESERVING_FNS.items():
        try:
            code = fn(baseline, entry)
        except Exception:
            tally[f"{name}:unavailable"] += 1
            continue
        if oracle.passes(code, entry, prob["inputs"], ref):
            record(name, code)
        else:
            # Generator bug or a genuinely behaviour-changing transform.
            tally[f"{name}:DROPPED_broke_behaviour"] += 1

    # Semantics-breaking: must fail.
    kept_mutants: dict[str, tuple[str, str]] = {}
    for name, fn in P.BREAKING_FNS.items():
        candidates = fn(baseline, config.MAX_MUTANT_CANDIDATES)
        if not candidates:
            tally[f"{name}:unavailable"] += 1
            continue
        for desc, code in candidates:
            if not oracle.passes(code, entry, prob["inputs"], ref):
                record(name, code, {"site": desc})
                kept_mutants[name] = (desc, code)
                break
        else:
            # Every candidate still passed: all sites are equivalent mutants.
            tally[f"{name}:all_equivalent"] += 1

    # The two docstring conditions.
    if docgen is not None:
        doc = docgen.misleading(baseline)
        if doc:
            code = P.set_docstring(baseline, entry, doc)
            if oracle.passes(code, entry, prob["inputs"], ref):
                record("misleading_doc", code)
            else:
                tally["misleading_doc:DROPPED_behaviour_changed"] += 1
        else:
            tally["misleading_doc:unavailable"] += 1

        chosen = next((m for m in MUTANT_PREFERENCE if m in kept_mutants), None)
        if chosen is None:
            tally["consistent_bug_doc:unavailable_no_mutant"] += 1
        else:
            desc, mutant_code = kept_mutants[chosen]
            doc = docgen.consistent_with_bug(mutant_code)
            if doc:
                code = P.set_docstring(mutant_code, entry, doc)
                if not oracle.passes(code, entry, prob["inputs"], ref):
                    record("consistent_bug_doc", code, {"from": chosen, "site": desc})
                else:
                    tally["consistent_bug_doc:DROPPED_now_passes"] += 1
            else:
                tally["consistent_bug_doc:unavailable"] += 1

    return out, tally


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N problems (pilot)")
    ap.add_argument("--max-inputs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-doc", action="store_true",
                    help="build the 8 code conditions only; no API credentials needed")
    ap.add_argument("--source", default="humaneval", choices=["humaneval", "lcb"],
                    help="humaneval (canonical solutions ship with the dataset) or "
                         "lcb (needs verified solutions from solve_lcb.py first)")
    ap.add_argument("--lcb-versions", nargs="*", default=["v6"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.source == "humaneval":
        problems = oracle.load_problems(max_inputs=args.max_inputs)
    else:
        from . import livecodebench
        problems = livecodebench.load_problems(versions=args.lcb_versions,
                                               max_inputs=args.max_inputs)
    args.out = args.out or str(config.DATA / f"dataset_{args.source}.jsonl")

    if args.limit:
        problems = problems[: args.limit]
    print(f"{len(problems)} problems from {args.source}, "
          f"up to {args.max_inputs} inputs each")

    docgen = None
    if not args.skip_doc:
        from .docgen import DocGenerator
        docgen = DocGenerator()

    records: list[dict] = []
    tally: Counter = Counter()
    failures = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_problem, p, docgen): p for p in problems}
        for i, fut in enumerate(as_completed(futures), start=1):
            prob = futures[fut]
            try:
                recs, t = fut.result()
            except Exception:
                failures += 1
                tally["problem_raised"] += 1
                print(f"  !! {prob['task_id']} raised:\n{traceback.format_exc(limit=3)}")
                continue
            records.extend(recs)
            tally.update(t)
            if i % 20 == 0 or i == len(problems):
                print(f"  {i}/{len(problems)} problems, {len(records)} variants", flush=True)

    records.sort(key=lambda r: (r["task_id"], r["condition"]))
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nwrote {len(records)} variants to {args.out}")
    print("\n--- availability report ---")
    for cond in config.ALL_CONDITIONS:
        print(f"  {cond:22s} {tally.get(cond, 0):4d}")
    print("\n--- dropped / unavailable ---")
    for k in sorted(k for k in tally if k not in config.ALL_CONDITIONS):
        print(f"  {k:42s} {tally[k]:4d}")
    if failures:
        print(f"\n{failures} problems raised during construction")


if __name__ == "__main__":
    main()
