"""Experiment configuration: datasets, conditions, models, run parameters."""

import json
from pathlib import Path


def cache_read(path: Path) -> dict | None:
    """Read a cache entry. None means treat it as a miss."""
    try:
        text = path.read_text()
    except OSError:
        return None
    try:
        blob = json.loads(text)
    except json.JSONDecodeError:
        # Truncated by an interrupted write. Drop it so the retry sticks.
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return blob if isinstance(blob, dict) else None


def cache_write(path: Path, payload: dict) -> bool:
    """Write a cache entry. The cache is an optimisation, so a failure here
    costs a repeated API call later rather than aborting the run."""
    try:
        path.write_text(json.dumps(payload))
        return True
    except OSError:
        return False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
for _d in (DATA, RESULTS, CACHE):
    _d.mkdir(exist_ok=True)


# Conditions. Preserving transforms must still pass the oracle, breaking ones
# must now fail it; build_dataset drops any variant that disagrees.
PRESERVING = ["rename", "strip_docstring", "reformat", "extract_constants"]
BREAKING = ["mut_compare", "mut_offbyone", "mut_swap"]

# The docstring 2x2. `baseline` is cell A and is also the reference point every
# other condition is differenced against.
#   A baseline            correct code, truthful docstring
#   B misleading_doc      correct code, docstring describes different behaviour
#   C mut_*               buggy code,   docstring still describes correct behaviour
#   D consistent_bug_doc  buggy code,   docstring rewritten to match the bug
DOC_CONDITIONS = ["misleading_doc", "consistent_bug_doc"]

ALL_CONDITIONS = ["baseline"] + PRESERVING + BREAKING + DOC_CONDITIONS

ORACLE_TRUTH = {c: True for c in ["baseline"] + PRESERVING + ["misleading_doc"]}
ORACLE_TRUTH.update({c: False for c in BREAKING + ["consistent_bug_doc"]})


from .providers import ModelSpec

# Model ids churn. Check what your endpoint actually serves before a full run:
#     python -m harness.list_models --contains qwen
OPENROUTER = dict(provider="openai",
                  base_url_env="OPENROUTER_BASE_URL",
                  base_url_default="https://openrouter.ai/api/v1",
                  api_key_env="OPENROUTER_API_KEY")

OLLAMA = dict(provider="openai",
              base_url_env="OLLAMA_BASE_URL",
              base_url_default="http://localhost:11434/v1",
              api_key_env=None)

SPECS: dict[str, ModelSpec] = {s.name: s for s in [
    # Open-weight judges.
    ModelSpec(name="qwen",     model="qwen/qwen3-next-80b-a3b-instruct", **OPENROUTER),
    ModelSpec(name="deepseek", model="deepseek/deepseek-v3.2-exp",       **OPENROUTER),
    ModelSpec(name="llama",    model="meta-llama/llama-3.3-70b-instruct", **OPENROUTER),

    # Local and free, for smoke tests only.
    ModelSpec(name="local", model="qwen2.5-coder:7b", **OLLAMA),

    # Dataset construction.
    ModelSpec(name="gemini", model="google/gemini-3.7-flash", **OPENROUTER),

    # Frontier judges, routed through OpenRouter like the others so the model
    # is the only thing that differs between arms. GPT was picked over Gemini
    # because gemini-3.7-flash writes the misleading docstrings, and a judge
    # scoring its own generated stimulus would confound the headline contrast.
    ModelSpec(name="claude", model="anthropic/claude-haiku-4.5",
              open_weight=False, **OPENROUTER),
    ModelSpec(name="gpt", model="openai/gpt-5.1",
              open_weight=False, **OPENROUTER),

    # Direct-to-Anthropic variants, need ANTHROPIC_API_KEY.
    ModelSpec(name="haiku",  provider="anthropic", model="claude-haiku-4-5",
              temperature=None, open_weight=False),
    ModelSpec(name="sonnet", provider="anthropic", model="claude-sonnet-5",
              temperature=None, open_weight=False),
    ModelSpec(name="opus",   provider="anthropic", model="claude-opus-5",
              temperature=None, open_weight=False),
]}

JUDGES = ["qwen", "deepseek", "llama"]

# The misleading docstring is the manipulation the headline result rests on, so
# generation stays on a strong model. It is only ~580 calls across both
# datasets. Hand-check twenty outputs before switching this to a weaker model.
BUILDER = "gemini"

# Judges run with thinking off: that is how they are deployed at scale, and it
# keeps the comparison across models clean. "adaptive" enables the thinking arm.
JUDGE_THINKING = "disabled"
JUDGE_EFFORT = "low"
JUDGE_MAX_TOKENS = 1024
REPS = 3
CONCURRENCY = 12

# The task specification is extracted once from the baseline docstring and held
# byte-identical across conditions. Without this, strip_docstring would confound
# a lost surface cue with a lost specification.
SHOW_SPEC = True

EXEC_TIMEOUT_S = 6.0          # per candidate, whole test suite
MAX_MUTANT_CANDIDATES = 12    # sites tried per breaking family
SEED = 20260825
