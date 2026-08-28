"""Judge calls: prompt, structured output, disk cache, concurrency.

A judge is just a ModelSpec from config.SPECS, so open-weight endpoints, local
Ollama and Claude all run through the same path into the same results file.

Every call is cached on disk under a key covering everything that could change
the answer, including provider, model id, temperature and rep index. Runs are
long and get interrupted, so only genuinely new cells are paid for.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Iterable

from . import config, providers

SYSTEM = (
    "You are reviewing a candidate Python implementation against a task "
    "specification. Judge only whether the implementation's runtime behaviour "
    "satisfies the specification. Report your assessment in the required "
    "format."
)

# Control for the misleading-docstring result. The objection it answers is that
# code whose docstring contradicts it really is defective, so rejecting it is
# defensible. This variant says outright that docstrings are unreliable and
# outside the specification, while stopping short of naming a verdict.
SYSTEM_IGNORE_DOCS = (
    "You are reviewing a candidate Python implementation against a task "
    "specification. Judge only whether the implementation's runtime behaviour "
    "satisfies the specification.\n\n"
    "The candidate's own docstrings and comments are unreliable annotations. "
    "They are not part of the specification and may be inaccurate or "
    "out of date. Disregard them: determine what the code actually does when "
    "executed, and compare that behaviour against the task specification "
    "above.\n\n"
    "Report your assessment in the required format."
)

SYSTEM_VARIANTS = {
    "default": SYSTEM,
    "ignore_docs": SYSTEM_IGNORE_DOCS,
}

USER_TEMPLATE = """\
## Task specification

{spec}

## Candidate implementation

```python
{code}
```

Assess whether this implementation satisfies the specification above."""

USER_TEMPLATE_NO_SPEC = """\
## Candidate implementation

```python
{code}
```

Assess whether this implementation is correct."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {
            "type": "boolean",
            "description": "True if the implementation satisfies the specification for all valid inputs.",
        },
        "confidence": {
            "type": "integer",
            "description": "Confidence in the correctness verdict, 0-100.",
        },
        "quality": {
            "type": "integer",
            "description": "Overall quality of the implementation, 1 (poor) to 5 (excellent).",
        },
        "reason": {
            "type": "string",
            "description": "One or two sentences justifying the verdict.",
        },
    },
    "required": ["correct", "confidence", "quality", "reason"],
    "additionalProperties": False,
}


def build_user(code: str, spec: str | None) -> str:
    if spec is not None and config.SHOW_SPEC:
        return USER_TEMPLATE.format(spec=spec.strip(), code=code.strip())
    return USER_TEMPLATE_NO_SPEC.format(code=code.strip())


@dataclass
class JudgeRequest:
    key: str            # caller-supplied identity, echoed back on the result
    judge: str          # name of a spec in config.SPECS
    code: str
    spec: str | None
    rep: int
    prompt_variant: str = "default"


@dataclass
class JudgeResult:
    key: str
    judge: str
    rep: int
    verdict: dict | None
    error: str | None
    usage: dict | None
    json_mode: str | None
    cached: bool
    prompt_variant: str = "default"


class JudgePool:
    def __init__(self, concurrency: int = config.CONCURRENCY):
        self.concurrency = concurrency
        self._providers: dict[str, object] = {}
        self._lock = threading.Lock()
        self._n_cached = 0
        self._n_called = 0
        self._modes: dict[str, int] = {}

    def _provider(self, name: str):
        with self._lock:
            if name not in self._providers:
                self._providers[name] = providers.build(config.SPECS[name])
            return self._providers[name]

    def _cache_key(self, req: JudgeRequest) -> str:
        spec = config.SPECS[req.judge]
        payload = json.dumps(
            {
                "spec": asdict(spec),
                "code": req.code,
                "task_spec": req.spec if config.SHOW_SPEC else None,
                "rep": req.rep,
                "system": SYSTEM_VARIANTS[req.prompt_variant],
                "schema": VERDICT_SCHEMA,
                "thinking": config.JUDGE_THINKING,
                "effort": config.JUDGE_EFFORT,
                "max_tokens": config.JUDGE_MAX_TOKENS,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _judge_one(self, req: JudgeRequest) -> JudgeResult:
        path = config.CACHE / f"{self._cache_key(req)}.json"
        blob = config.cache_read(path) if path.exists() else None
        if blob is not None:
            with self._lock:
                self._n_cached += 1
            return JudgeResult(req.key, req.judge, req.rep, blob.get("verdict"),
                               blob.get("error"), blob.get("usage"),
                               blob.get("json_mode"), cached=True,
                               prompt_variant=req.prompt_variant)

        out = self._provider(req.judge).complete(
            system=SYSTEM_VARIANTS[req.prompt_variant],
            user=build_user(req.code, req.spec),
            schema=VERDICT_SCHEMA,
            max_tokens=config.JUDGE_MAX_TOKENS,
            thinking=config.JUDGE_THINKING,
            effort=config.JUDGE_EFFORT,
        )

        # Only successes are cached. Caching a 402/429/timeout would make the
        # failure permanent: the rerun reads it back instead of retrying.
        if out.data is not None:
            config.cache_write(path, {"verdict": out.data, "error": None,
                                      "usage": out.usage, "json_mode": out.json_mode,
                                      "judge": req.judge, "rep": req.rep})
        with self._lock:
            self._n_called += 1
            if out.json_mode:
                self._modes[out.json_mode] = self._modes.get(out.json_mode, 0) + 1
        return JudgeResult(req.key, req.judge, req.rep, out.data, out.error,
                           out.usage, out.json_mode, cached=False,
                           prompt_variant=req.prompt_variant)

    def run(self, requests: Iterable[JudgeRequest], progress_every: int = 200) -> list[JudgeResult]:
        requests = list(requests)
        results: list[JudgeResult] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._judge_one, r) for r in requests]
            for i, fut in enumerate(as_completed(futures), start=1):
                results.append(fut.result())
                if i % progress_every == 0 or i == len(requests):
                    print(f"  {i}/{len(requests)}  "
                          f"(cache {self._n_cached}, api {self._n_called})", flush=True)
        if self._modes:
            print(f"  structured-output modes used: {self._modes}")
        return results


def preflight(names: list[str] | None = None) -> None:
    """Call every judge once on a trivial input before a long run, and report
    the structured-output mode each endpoint accepted. A judge stuck in prompt
    mode is worth knowing about now rather than afterwards."""
    for name in (names or config.JUDGES):
        spec = config.SPECS[name]
        try:
            out = providers.build(spec).complete(
                system=SYSTEM,
                user=build_user("def add(a, b):\n    return a + b", "Return the sum of a and b."),
                schema=VERDICT_SCHEMA,
                max_tokens=config.JUDGE_MAX_TOKENS,
                thinking=config.JUDGE_THINKING,
                effort=config.JUDGE_EFFORT,
            )
            status = "OK " if out.data else "FAIL"
            detail = out.data if out.data else out.error
            print(f"  {status} {name:10s} {spec.model:45s} mode={out.json_mode} {str(detail)[:110]}")
        except Exception as e:
            print(f"  FAIL {name:10s} {spec.model:45s} {type(e).__name__}: {str(e)[:110]}")
