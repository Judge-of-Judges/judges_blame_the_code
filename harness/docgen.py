"""Generation of the two adversarial docstring conditions.

These are the only parts of the dataset that need a model to build, and they
carry the paper's headline contrast:

  misleading_doc     correct code + a confident docstring describing *different*
                     behaviour. Ground truth: the code is correct.
  consistent_bug_doc buggy code + a docstring that accurately describes what the
                     buggy code does. Ground truth: the code is incorrect, since
                     it still violates the task specification.

A judge that reads behaviour scores these on the code. A judge that reads
plausibility scores them on whether prose and code agree, which is backwards on
both conditions.

Generation is cached on disk keyed by the source it was derived from, so the
docstrings are stable across reruns and the dataset is reproducible.
"""

from __future__ import annotations

import hashlib
import json

from . import config, providers

_MISLEADING_SYSTEM = (
    "You write documentation for Python functions. You write in the exact "
    "style of the surrounding codebase and you always sound authoritative."
)

_MISLEADING_USER = """\
Below is a Python function with its true docstring.

```python
{code}
```

Write a REPLACEMENT docstring for this function that confidently describes \
*different* behaviour from what the code actually does. Change something \
substantive and checkable -- the return convention, an edge-case rule, the \
handling of empty or boundary input, or the ordering of results. Keep the same \
prose style, length, and formatting conventions as the original, and if the \
original contains doctest examples, include examples in the same format that \
are consistent with your (different) description.

The docstring must read as completely ordinary. Do not hedge, do not hint that \
anything is unusual, and do not mention the code."""

_CONSISTENT_SYSTEM = (
    "You write documentation for Python functions. You describe precisely what "
    "the code in front of you does, without editorialising."
)

_CONSISTENT_USER = """\
Below is a Python function.

```python
{code}
```

Write a docstring that accurately and confidently describes what this code \
ACTUALLY does when executed, including its behaviour on edge cases. Match the \
prose style, length, and formatting of a normal docstring for this codebase, \
and if it is natural to include doctest examples, make them consistent with \
the code's real behaviour.

Do not mention bugs, correctness, limitations, or the possibility that the \
behaviour is unintended. Write it as the plain documentation of intended \
behaviour."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "docstring": {
            "type": "string",
            "description": "The docstring body, without surrounding triple quotes.",
        }
    },
    "required": ["docstring"],
    "additionalProperties": False,
}


class DocGenerator:
    """Runs on config.BUILDER by default.

    This is the instrument rather than the object of study. A weak generator
    that hedges or writes transparently wrong descriptions attenuates the effect
    and makes a null result uninterpretable, and at roughly 580 calls across
    both datasets it is the wrong place to economise. If you do point it at an
    open model, hand-check twenty outputs first."""

    def __init__(self, spec_name: str | None = None):
        self.spec_name = spec_name or config.BUILDER
        self.spec = config.SPECS[self.spec_name]
        self.provider = providers.build(self.spec)

    def _cached(self, kind: str, code: str) -> str | None:
        digest = hashlib.sha256(
            json.dumps({"kind": kind, "code": code, "model": self.spec.model,
                        "provider": self.spec.provider}, sort_keys=True).encode()
        ).hexdigest()
        path = config.CACHE / f"doc-{digest}.json"
        cached = config.cache_read(path) if path.exists() else None
        if cached is not None:
            return cached.get("docstring")

        system, template = (
            (_MISLEADING_SYSTEM, _MISLEADING_USER)
            if kind == "misleading"
            else (_CONSISTENT_SYSTEM, _CONSISTENT_USER)
        )
        out = self.provider.complete(
            system=system,
            user=template.format(code=code.strip()),
            schema=_SCHEMA,
            max_tokens=2048,
            thinking="disabled",
            effort="medium",
        )
        doc = out.data["docstring"] if out.data else None
        # Only successes are cached. A 402, 429 or truncated response is
        # transient, and caching it would make the failure permanent.
        if doc:
            config.cache_write(path, {"docstring": doc})
        return doc

    def misleading(self, correct_code: str) -> str | None:
        """A confident docstring describing behaviour the correct code does not have."""
        return self._cached("misleading", correct_code)

    def consistent_with_bug(self, buggy_code: str) -> str | None:
        """A docstring that truthfully describes what the buggy code does."""
        return self._cached("consistent", buggy_code)
