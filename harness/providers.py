"""Anthropic and OpenAI-compatible endpoints behind one interface.

The OpenAI adapter covers hosted open-weight providers, self-hosted vLLM and
local Ollama, since they all speak chat-completions; only base_url, api_key and
model differ.

Support for `response_format` varies by provider and by model, so each
(endpoint, model) pair is probed once and pinned to the strictest mode it
accepts:

    json_schema  constrained decoding, schema enforced
    json_object  valid JSON, shape not guaranteed; schema goes in the prompt
    prompt       nothing guaranteed; JSON extracted from free text

The mode is recorded on every result, since the fraction of verdicts recovered
by extraction belongs in the methods section. `coerce()` normalises the loose
types open models return ("true" for a boolean, "4" or 4.0 for an integer) so a
judge is not scored as failed over formatting.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to call one model, and to name it in the results."""
    name: str                       # short label used in every output file
    provider: str                   # "anthropic" | "openai"
    model: str                      # provider-side model id
    base_url_env: str | None = None # env var holding the endpoint
    base_url_default: str | None = None
    api_key_env: str | None = None  # env var holding the key
    temperature: float | None = 0.0 # ignored by the anthropic provider
    json_mode: str = "auto"         # auto | json_schema | json_object | prompt
    open_weight: bool = True        # for the paper's model table

    @property
    def base_url(self) -> str | None:
        if self.base_url_env and os.environ.get(self.base_url_env):
            return os.environ[self.base_url_env]
        return self.base_url_default

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        # Local servers ignore the key but the OpenAI client insists on one.
        return os.environ.get(self.api_key_env) or "not-needed"


@dataclass
class Completion:
    data: dict | None
    error: str | None
    usage: dict | None
    json_mode: str | None = None     # which mode actually produced this
    raw: str | None = field(default=None, repr=False)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Best-effort recovery of one JSON object from arbitrary model output."""
    if not text:
        return None
    for candidate in ([m.group(1) for m in _FENCE.finditer(text)] + [text]):
        candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # Scan for the first balanced {...}, respecting strings and escapes.
        depth = 0
        start = None
        in_str = False
        esc = False
        for i, ch in enumerate(candidate):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(candidate[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        start = None
    return None


_TRUE = {"true", "yes", "y", "1", "correct", "pass", "passes"}
_FALSE = {"false", "no", "n", "0", "incorrect", "fail", "fails"}


def coerce(obj: dict, schema: dict) -> dict | None:
    """Normalise types against the schema. Returns None if a required field is
    missing or cannot be coerced."""
    props = schema.get("properties", {})
    out: dict[str, Any] = {}
    for key, spec in props.items():
        if key not in obj:
            continue
        val = obj[key]
        want = spec.get("type")
        try:
            if want == "boolean":
                if isinstance(val, bool):
                    out[key] = val
                elif isinstance(val, (int, float)):
                    out[key] = bool(val)
                elif isinstance(val, str):
                    low = val.strip().lower()
                    if low in _TRUE:
                        out[key] = True
                    elif low in _FALSE:
                        out[key] = False
                    else:
                        return None
                else:
                    return None
            elif want == "integer":
                out[key] = int(round(float(val)))
            elif want == "number":
                out[key] = float(val)
            elif want == "string":
                out[key] = val if isinstance(val, str) else json.dumps(val)
            else:
                out[key] = val
        except (TypeError, ValueError):
            return None
    for req in schema.get("required", []):
        if req not in out:
            return None
    return out


def schema_instruction(schema: dict) -> str:
    return (
        "\n\nRespond with a single JSON object and nothing else — no prose, no "
        "markdown fence. It must match this schema exactly:\n"
        + json.dumps(schema, indent=2)
    )


class AnthropicProvider:
    def __init__(self, spec: ModelSpec, max_retries: int = 8, timeout: float = 180.0):
        import anthropic
        self.spec = spec
        kwargs: dict[str, Any] = {"max_retries": max_retries, "timeout": timeout}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        if spec.api_key_env and os.environ.get(spec.api_key_env):
            kwargs["api_key"] = os.environ[spec.api_key_env]
        self.client = anthropic.Anthropic(**kwargs)

    def complete(self, system: str, user: str, schema: dict, max_tokens: int,
                 thinking: str = "disabled", effort: str = "low",
                 temperature: float | None = None) -> Completion:
        # temperature is accepted and ignored: current Claude models reject it
        # with a 400, and callers get variation from sampling instead.
        kwargs: dict[str, Any] = dict(
            model=self.spec.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort,
                           "format": {"type": "json_schema", "schema": schema}},
            thinking={"type": "disabled"} if thinking == "disabled" else {"type": "adaptive"},
        )
        try:
            resp = self.client.messages.create(**kwargs)
        except Exception as e:
            return Completion(None, f"{type(e).__name__}: {e}"[:400], None)

        usage = {"input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens}
        if resp.stop_reason == "refusal":
            return Completion(None, "refusal", usage, "json_schema")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if not text:
            return Completion(None, f"no text (stop_reason={resp.stop_reason})", usage, "json_schema")
        data = coerce(extract_json(text) or {}, schema)
        if data is None:
            return Completion(None, "schema coercion failed", usage, "json_schema", text[:600])
        return Completion(data, None, usage, "json_schema")


class OpenAICompatProvider:
    """Any OpenAI chat-completions endpoint: hosted open-weight providers,
    self-hosted vLLM, or local Ollama."""

    # Discovered mode per (base_url, model), shared across threads so 25k calls
    # don't each rediscover that json_schema is rejected.
    _modes: dict[tuple, str] = {}
    _fails: dict[tuple, int] = {}
    _lock = threading.Lock()

    # A model that occasionally returns junk is not a model that cannot honour a
    # schema, so demotion needs repeated evidence rather than one bad response.
    DEMOTE_AFTER = 6

    def __init__(self, spec: ModelSpec, max_retries: int = 6, timeout: float = 180.0):
        from openai import OpenAI
        self.spec = spec
        self.client = OpenAI(
            base_url=spec.base_url or "https://openrouter.ai/api/v1",
            api_key=spec.api_key or "not-needed",
            max_retries=max_retries,
            timeout=timeout,
        )

    @property
    def _mode_key(self):
        return (self.spec.base_url, self.spec.model)

    def _current_mode(self) -> str:
        if self.spec.json_mode != "auto":
            return self.spec.json_mode
        with self._lock:
            return self._modes.get(self._mode_key, "json_schema")

    @staticmethod
    def _next_rung(mode: str) -> str:
        return {"json_schema": "json_object", "json_object": "prompt"}.get(mode, "prompt")

    def _demote_sticky(self, failed: str) -> None:
        """Pin the endpoint one rung lower for the rest of the run."""
        with self._lock:
            self._modes[self._mode_key] = self._next_rung(failed)
            self._fails[self._mode_key] = 0

    def _note_failure(self) -> int:
        with self._lock:
            n = self._fails.get(self._mode_key, 0) + 1
            self._fails[self._mode_key] = n
            return n

    def _note_success(self) -> None:
        with self._lock:
            self._fails[self._mode_key] = 0

    @staticmethod
    def _mode_unsupported(err: Exception) -> bool:
        """Did the endpoint reject the parameter, as opposed to failing for a
        transient reason? Only the former justifies a permanent demotion."""
        s = str(err).lower()
        return ("response_format" in s or "json_schema" in s
                or "not supported" in s or "unsupported" in s)

    def _request(self, system: str, user: str, schema: dict, max_tokens: int, mode: str,
                 temperature: float | None = None):
        kwargs: dict[str, Any] = dict(
            model=self.spec.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        temp = self.spec.temperature if temperature is None else temperature
        if temp is not None:
            kwargs["temperature"] = temp
        if mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "strict": True, "schema": schema},
            }
        elif mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["messages"][-1]["content"] += schema_instruction(schema)
        else:
            kwargs["messages"][-1]["content"] += schema_instruction(schema)
        return self.client.chat.completions.create(**kwargs)

    def complete(self, system: str, user: str, schema: dict, max_tokens: int,
                 thinking: str = "disabled", effort: str = "low",
                 temperature: float | None = None) -> Completion:
        sticky = self._current_mode()
        mode = sticky
        last_err = None
        first_try = True

        # Walk down the ladder within this call to salvage a verdict, kept
        # separate from the endpoint's pinned mode. At temperature 0 the
        # response is deterministic, so each rung is tried once.
        for _ in range(3):
            try:
                resp = self._request(system, user, schema, max_tokens, mode, temperature)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"[:400]
                if self._mode_unsupported(e) and mode != "prompt":
                    self._demote_sticky(mode)   # endpoint genuinely rejects it
                    mode = self._next_rung(mode)
                    continue
                # Transient (402/429/5xx/timeout): do not touch sticky state.
                return Completion(None, last_err, None, mode)

            usage = None
            if getattr(resp, "usage", None):
                usage = {"input_tokens": resp.usage.prompt_tokens,
                         "output_tokens": resp.usage.completion_tokens}
            text = resp.choices[0].message.content if resp.choices else None
            data = coerce(extract_json(text or "") or {}, schema)
            if data is not None:
                if first_try:
                    self._note_success()
                return Completion(data, None, usage, mode)

            last_err = "unparseable or schema-invalid output"
            if first_try and mode == sticky:
                # Flakiness rather than incapacity: demote only once failures
                # at this rung are persistent.
                if self._note_failure() >= self.DEMOTE_AFTER and mode != "prompt":
                    self._demote_sticky(mode)
                first_try = False
            if mode == "prompt":
                return Completion(None, last_err, usage, mode, (text or "")[:600])
            mode = self._next_rung(mode)
        return Completion(None, last_err or "exhausted modes", None, mode)


def build(spec: ModelSpec):
    if spec.provider == "anthropic":
        return AnthropicProvider(spec)
    if spec.provider == "openai":
        return OpenAICompatProvider(spec)
    raise ValueError(f"unknown provider {spec.provider!r}")


def list_models(base_url_env: str, api_key_env: str, contains: str = "") -> list[str]:
    """Ask an endpoint what it serves, so model ids are looked up not guessed."""
    from openai import OpenAI
    client = OpenAI(base_url=os.environ[base_url_env],
                    api_key=os.environ.get(api_key_env) or "not-needed")
    ids = sorted(m.id for m in client.models.list().data)
    return [i for i in ids if contains.lower() in i.lower()]
