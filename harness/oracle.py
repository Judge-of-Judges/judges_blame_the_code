"""Execution oracle: differential testing against the canonical solution.

    canonical_solution(x) vs candidate(x)  for every x in the test inputs

A candidate passes only if it agrees with the canonical solution on every
input, including on which inputs raise. That is stronger than checking assert
statements and it is what kills equivalent mutants.

Each program runs in a fresh subprocess under a wall-clock timeout, so an
infinite loop introduced by a mutation costs one timeout rather than the run.
Results are canonicalised before comparison, keeping the type tag so a
candidate returning (1, 2) against a reference [1, 2] counts as disagreement.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config

# Defined once as source and used twice: exec'd into this module for
# parent-side comparison, and embedded verbatim in the child runner below.
_CANON_SRC = r'''
def canon(x, nd=6):
    if isinstance(x, float):
        if math.isnan(x):  return ["float", "nan"]
        if math.isinf(x):  return ["float", "inf" if x > 0 else "-inf"]
        return ["float", round(x, nd)]
    if isinstance(x, bool):          return ["bool", x]
    if isinstance(x, int):           return ["int", x]
    if isinstance(x, str):           return ["str", x]
    if x is None:                    return ["none"]
    if isinstance(x, tuple):         return ["tuple", [canon(v, nd) for v in x]]
    if isinstance(x, list):          return ["list", [canon(v, nd) for v in x]]
    if isinstance(x, (set, frozenset)):
        return ["set", sorted([json.dumps(canon(v, nd)) for v in x])]
    if isinstance(x, dict):
        return ["dict", sorted([[json.dumps(canon(k, nd)), canon(v, nd)] for k in x for v in [x[k]]])]
    return ["repr", repr(x)]
'''
exec(_CANON_SRC, globals())

# The child process. Reads {code, entry_point, inputs} on stdin and writes one
# canonicalised result per input on stdout. It must run standalone, so nothing
# here may import from the harness.
_RUNNER = "import json, math, sys\n" + r'''
# Factorial and fibonacci problems return integers with thousands of digits.
# Python 3.11+ refuses to str() an int over 4300 digits by default, which
# surfaces as a JSON encoding crash rather than a wrong answer.
sys.set_int_max_str_digits(200000)
''' + _CANON_SRC + r'''
def resolve(ns, name):
    """Module-level function, or a method on a class (LeetCode-style
    `class Solution: def foo(self, ...)`), which is how LiveCodeBench's
    functional problems are shaped."""
    if name in ns and callable(ns[name]):
        return ns[name]
    for value in ns.values():
        if isinstance(value, type) and callable(getattr(value, name, None)):
            return getattr(value(), name)
    raise KeyError(name)

payload = json.loads(sys.stdin.read())
ns = {"__name__": "__candidate__"}
try:
    exec(compile(payload["code"], "<candidate>", "exec"), ns)
    fn = resolve(ns, payload["entry_point"])
except BaseException as e:
    print(json.dumps({"load_error": type(e).__name__}))
    sys.exit(0)

out = []
for args in payload["inputs"]:
    try:
        out.append(canon(fn(*args)))
    except BaseException as e:
        out.append(["raised", type(e).__name__])
print(json.dumps({"results": out}))
'''


class OracleError(RuntimeError):
    pass


def _run(code: str, entry_point: str, inputs: list, timeout: float) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_RUNNER)
        runner_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, runner_path],
            input=json.dumps({"code": code, "entry_point": entry_point, "inputs": inputs}),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    finally:
        Path(runner_path).unlink(missing_ok=True)

    if proc.returncode != 0 or not proc.stdout.strip():
        return {"crash": (proc.stderr or "")[-400:]}
    try:
        return json.loads(proc.stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return {"crash": "unparseable runner output"}


def reference_outputs(canonical_code: str, entry_point: str, inputs: list,
                      timeout: float = config.EXEC_TIMEOUT_S) -> list:
    """Run the reference solution once; its outputs become ground truth."""
    res = _run(canonical_code, entry_point, inputs, timeout)
    if "results" not in res:
        raise OracleError(f"reference solution did not run cleanly: {res}")
    return res["results"]


def passes(candidate_code: str, entry_point: str, inputs: list, reference: list,
           timeout: float = config.EXEC_TIMEOUT_S) -> bool:
    """True if the candidate agrees with the reference on every input.

    Timeouts, import-time errors and crashes all count as failure: an infinite
    loop is a real bug, not an inconclusive result."""
    res = _run(candidate_code, entry_point, inputs, timeout)
    if "results" not in res:
        return False
    return res["results"] == reference


def load_problems(max_inputs: int = 200) -> list[dict]:
    """HumanEval+ problems with a capped, deterministic input set.

    A couple of hundred inputs already detects essentially every
    non-equivalent single-token mutation, and capping keeps the oracle pass to
    minutes. Inputs are taken in dataset order (base then plus) so the
    selection is reproducible."""
    from evalplus.data import get_human_eval_plus

    problems = []
    for task_id, p in get_human_eval_plus().items():
        inputs = list(p["base_input"]) + list(p.get("plus_input", []))
        problems.append({
            "task_id": task_id,
            "entry_point": p["entry_point"],
            "prompt": p["prompt"],
            "canonical": p["prompt"] + p["canonical_solution"],
            "inputs": inputs[:max_inputs],
            "n_inputs_total": len(inputs),
        })
    return problems
