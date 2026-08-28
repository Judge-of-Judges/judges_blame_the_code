"""LiveCodeBench loader, a second and structurally different code distribution.

This is a generalisation check rather than a contamination control. LCB v6
covers contests from 2025-01-04 to 2025-04-06, about thirteen months before the
judges' May 2026 cutoff, so these problems are as memorisable as HumanEval.
What they buy is LeetCode contest code: class Solution methods, 20-60 line
bodies, real algorithmic difficulty.

Two structural differences from HumanEval are handled here. LCB ships no
reference solutions, so solve_lcb.py generates and execution-verifies them into
data/lcb_solutions.jsonl, and only problems whose solution passes every public
and private test become usable. LCB also has no docstring: the specification
lives in question_content and is inserted as the entry method's docstring so
every downstream condition behaves as it does on HumanEval.

Only functional problems are used, meaning those with starter_code. LCB's
stdin/stdout problems have no fn(*args) signature and do not fit the oracle.
"""

from __future__ import annotations

import ast
import base64
import json
import pickle
import zlib

from . import perturb as P

# Each release file is an increment, so vN is the concatenation of files 1..N.
# Dates are the contest windows each increment covers.
INCREMENTS = {
    "v1": "test.jsonl",    # May 2023 - Mar 2024
    "v2": "test2.jsonl",   # -> May 2024
    "v3": "test3.jsonl",   # -> Jul 2024
    "v4": "test4.jsonl",   # -> Sep 2024
    "v5": "test5.jsonl",   # -> Jan 2025
    "v6": "test6.jsonl",   # -> Apr 2025  (latest release)
}

REPO = "livecodebench/code_generation_lite"

# Imports LeetCode starter code assumes. Shown to the judge in every variant,
# so it is constant across conditions but not free; kept to three lines.
PREAMBLE = (
    "from typing import List, Optional, Tuple, Dict, Set\n"
    "import math, collections, heapq, bisect, itertools, functools, re\n"
    "from collections import defaultdict, Counter, deque\n"
)


def _parse_value(text: str):
    """LCB encodes each argument as a JSON scalar/array on its own line."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text                       # a bare unquoted string


def _decode_private(blob: str) -> list[dict]:
    """LCB stores private tests as base64(zlib(pickle(json_string)))."""
    if not blob:
        return []
    try:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(blob.encode()))))
    except Exception:
        try:
            return json.loads(blob)
        except Exception:
            return []


def _tests(row: dict) -> list[dict]:
    pub = json.loads(row["public_test_cases"]) if row.get("public_test_cases") else []
    return list(pub) + _decode_private(row.get("private_test_cases", ""))


def load_raw(versions: list[str] | None = None, since: str | None = None,
             max_inputs: int = 200) -> list[dict]:
    """Functional LCB problems with parsed inputs and expected outputs.

    No canonical field yet: these are unusable until solve_lcb.py has supplied
    a verified reference solution."""
    from huggingface_hub import hf_hub_download

    versions = versions or ["v6"]
    rows: list[dict] = []
    for v in versions:
        path = hf_hub_download(REPO, INCREMENTS[v], repo_type="dataset")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                r["_version"] = v
                rows.append(r)

    problems: list[dict] = []
    for r in rows:
        if not r.get("starter_code", "").strip():
            continue                       # stdin/stdout form; no fn(*args)
        if since and r.get("contest_date", "") < since:
            continue
        try:
            func_name = json.loads(r["metadata"])["func_name"]
        except (KeyError, json.JSONDecodeError):
            continue

        cases = [t for t in _tests(r) if t.get("testtype") == "functional"]
        if not cases:
            continue

        inputs, expected = [], []
        for t in cases:
            args = [_parse_value(line) for line in t["input"].split("\n") if line.strip() != ""]
            inputs.append(args)
            expected.append(_parse_value(t["output"]))

        problems.append({
            "task_id": f"LCB/{r['question_id']}",
            "entry_point": func_name,
            "question_title": r["question_title"],
            "question_content": r["question_content"],
            "starter_code": r["starter_code"],
            "contest_date": r.get("contest_date", ""),
            "difficulty": r.get("difficulty", ""),
            "inputs": inputs[:max_inputs],
            "expected": expected[:max_inputs],
            "n_inputs_total": len(inputs),
        })
    return problems


def load_problems(solutions_path=None, versions: list[str] | None = None,
                  since: str | None = None, max_inputs: int = 200,
                  max_spec_chars: int | None = 4000) -> list[dict]:
    """Problems in the same shape `oracle.load_problems` returns, so
    `build_dataset` consumes them unchanged.

    Requires `data/lcb_solutions.jsonl` from `solve_lcb.py`."""
    from . import config

    solutions_path = solutions_path or (config.DATA / "lcb_solutions.jsonl")
    try:
        with open(solutions_path) as f:
            sols = {
                r["task_id"]: r["code"]
                for r in map(json.loads, filter(str.strip, f))
                if r.get("verified")
            }
    except FileNotFoundError:
        raise SystemExit(
            f"{solutions_path} not found. Generate verified reference solutions first:\n"
            f"    python -m harness.solve_lcb --versions v6"
        )

    out: list[dict] = []
    for p in load_raw(versions=versions, since=since, max_inputs=max_inputs):
        code = sols.get(p["task_id"])
        if not code:
            continue

        # The statement becomes the entry method's docstring so every
        # docstring condition works identically to HumanEval.
        spec = p["question_content"]
        if max_spec_chars and len(spec) > max_spec_chars:
            spec = spec[:max_spec_chars].rstrip() + "\n[statement truncated]"
        try:
            canonical = P.set_docstring(PREAMBLE + code, p["entry_point"], spec)
        except (SyntaxError, KeyError):
            continue                       # solution does not define the entry point

        out.append({
            "task_id": p["task_id"],
            "entry_point": p["entry_point"],
            "prompt": p["starter_code"],
            "canonical": canonical,
            "inputs": p["inputs"],
            "n_inputs_total": p["n_inputs_total"],
            "difficulty": p["difficulty"],
            "contest_date": p["contest_date"],
        })
    return out
