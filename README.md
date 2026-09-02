# When the Docstring Lies, LLM Judges Blame the Code

Reproduction package for the JUDGe 2026 workshop paper (NeurIPS).

## Data

All 50,750 judge verdicts, joined to the oracle-verified program variants they
were given, are on the Hugging Face Hub:

**https://huggingface.co/datasets/JudgeOfJudges/judges_blame_the_code**

```python
from datasets import load_dataset

he = load_dataset("JudgeOfJudges/judges_blame_the_code", "HumanEvalP", split="train")
lcb = load_dataset("JudgeOfJudges/judges_blame_the_code", "LiveCodeBenchV6", split="train")
```

Each row is one verdict and carries both what the judge said (`said_correct`,
`confidence`, `quality`, `reason`) and what execution says is true
(`oracle_pass`), so the data works as a labelled benchmark on its own. The
verdict and variant files this was built from are also in `results/` and `data/`
here.

## TL;DR

We asked five LLM judges whether a Python function satisfies a task
specification. We verified every candidate program by execution first, so the
ground truth comes from an oracle rather than from human preference labels.

The judges turn out to be robust to appearance. Renaming every identifier,
reformatting, or deleting the docstring outright moves the verdict by at most
0.08. Replacing the docstring with one that describes different behaviour is a
different story: correct code that passes every test is then rejected, with a
paired drop of 0.25 to 0.82 in P(correct) in all ten judge-by-dataset cells.

So the judges are not leaning on the docstring for information. Removing it
costs nothing; contradicting the code with it costs about as much as a real
single-token bug. Telling judges directly that docstrings are unreliable and
should be disregarded closes only part of the gap, and their own one-sentence
justifications name the docstring in 69% of these rejections against under 1%
elsewhere.

## Results

Paired change in P(judge says "correct") against each problem's own baseline:

| Condition | HumanEval+ | LiveCodeBench |
|---|---|---|
| remove docstring | -0.006 to -0.043 | +0.013 to -0.076 |
| false docstring | -0.408 to -0.824 | -0.253 to -0.541 |
| flipped comparison (a real bug) | -0.501 to -0.752 | -0.211 to -0.438 |

Scale of the run:

| | |
|---|---|
| HumanEval+ | 161 problems, 1,453 oracle-verified variants |
| LiveCodeBench v4-v6 | 157 problems, 1,492 oracle-verified variants |
| Verdicts | 50,750, of which 87.1% schema-constrained |
| Judges | Qwen3-Next-80B, DeepSeek-V3.2, Llama-3.3-70B, Claude Haiku 4.5, GPT-5.1 |
| Repetitions | 3 per (problem, condition, judge) |

## Design

Every variant is built from one normalized baseline per problem,
`ast.unparse(ast.parse(original))`, so a judge comparing the baseline against
any condition sees exactly one intended difference and no incidental formatting
noise.

| Condition | Edit | Oracle |
|---|---|---|
| `baseline` | none | passes |
| `rename` | identifiers become `v0, v1, ...` | passes |
| `strip_docstring` | docstring removed | passes |
| `reformat` | indentation and blank lines | passes |
| `extract_constants` | literals hoisted to named locals | passes |
| `mut_compare` | comparison operator flipped | fails |
| `mut_offbyone` | integer literal shifted by 1 | fails |
| `mut_swap` | operand or argument order swapped | fails |
| `misleading_doc` | correct code, docstring describes other behaviour | passes |
| `consistent_bug_doc` | buggy code, docstring describes the bug | fails |

A variant reaches the dataset only if execution agrees with the oracle verdict
its condition declares. Preserving transforms that change behaviour and mutants
that turn out to be equivalent are dropped and counted, never silently kept.

The judge always sees a task specification, taken from the baseline docstring
and held byte-identical across every condition, plus the candidate code. Without
that, `strip_docstring` would confound a lost surface cue with a lost
specification.

## Layout

```
harness/config.py          datasets, conditions, models, run parameters
harness/perturb.py         AST and text transforms
harness/oracle.py          differential testing against the canonical solution
harness/docgen.py          the two adversarial docstring conditions
harness/build_dataset.py   assembly, oracle verification, availability report
harness/livecodebench.py   LCB loader
harness/solve_lcb.py       generates and verifies LCB reference solutions
harness/run_judges.py      fan-out over variants x judges x reps
harness/judges.py          prompt, structured output, disk cache, concurrency
harness/providers.py       Anthropic and OpenAI-compatible endpoints
harness/analyze.py         validity index, docstring 2x2, recency analysis
harness/compare_prompts.py the instruction control
harness/reasons.py         what judges cite when they reject
harness/paper_figs.py      Figure 1
harness/paper_appendix.py  appendix tables and Figure 2
harness/inspect_docs.py    manual review of the generated docstrings
harness/list_models.py     what a given endpoint actually serves

data/dataset_humaneval.jsonl   1,453 verified variants
data/dataset_lcb.jsonl         1,492 verified variants
data/lcb_solutions.jsonl       verified LCB reference solutions
results/verdicts_*.jsonl       every judge verdict, one JSON object per line
```

## Install

Python 3.11 or newer.

```bash
pip install openai anthropic pandas numpy scipy matplotlib evalplus huggingface_hub
```

## Reproducing the analysis without any API calls

The verdict files are included, so every number and figure in the paper can be
regenerated offline. This is the quickest way to check the results.

```bash
python -m harness.analyze --verdicts results/verdicts_humaneval_all.jsonl
```

```bash
python -m harness.analyze --verdicts results/verdicts_lcb_all.jsonl
```

The instruction control, comparing the default prompt against the one that tells
judges to disregard docstrings:

```bash
python -m harness.compare_prompts --a results/verdicts_humaneval_all.jsonl --b results/verdicts_ignoredocs_all.jsonl
```

What judges cite when they reject:

```bash
python -m harness.reasons
```

Paper figures and tables:

```bash
python -m harness.paper_figs && python -m harness.paper_appendix
```

Note that `results/verdicts_ignoredocs_frontier.jsonl` is the frontier-judge
component of `verdicts_ignoredocs_all.jsonl` and is contained in it. Use the
`_all` files for anything you want to count.

## Reproducing from scratch

This costs money and takes a few hours. Set your key first. Any
OpenAI-compatible endpoint works, including OpenRouter, Together, Groq,
Fireworks, DeepInfra, self-hosted vLLM and local Ollama.

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Model ids churn between releases. Check what your endpoint actually serves and
edit `config.SPECS` before a full run:

```bash
python -m harness.list_models --contains qwen
```

```bash
python -c "from harness import judges; judges.preflight()"
```

### HumanEval+ arm

The eight code conditions need no API access at all, since HumanEval+ ships
canonical solutions:

```bash
python -m harness.build_dataset --skip-doc --limit 20
```

The full build adds the two docstring conditions and needs the generator model:

```bash
python -m harness.build_dataset
```

Check the generated docstrings by hand before spending anything on judging. The
oracle can confirm the code still passes, but not that a docstring is
convincingly misleading:

```bash
python -m harness.inspect_docs --n 8
```

Then judge. `--dry-run` prints the call budget and stops:

```bash
python -m harness.run_judges --dataset data/dataset_humaneval.jsonl --dry-run
```

```bash
python -m harness.run_judges --dataset data/dataset_humaneval.jsonl --judges qwen deepseek llama claude gpt --out results/verdicts_humaneval_all.jsonl
```

### LiveCodeBench arm

LCB ships no reference solutions, so they are generated and kept only if they
pass every public and private test:

```bash
python -m harness.solve_lcb --versions v4 v5 v6 --attempts 3
```

```bash
python -m harness.build_dataset --source lcb --lcb-versions v4 v5 v6
```

```bash
python -m harness.run_judges --dataset data/dataset_lcb.jsonl --judges qwen deepseek llama claude gpt --out results/verdicts_lcb_all.jsonl
```

### Instruction control

Same variants, different system prompt:

```bash
python -m harness.run_judges --dataset data/dataset_humaneval.jsonl --prompt-variant ignore_docs --conditions baseline misleading_doc mut_compare --out results/verdicts_ignoredocs_all.jsonl
```

### Free local smoke test

Slow, and fine for twenty problems rather than a full run:

```bash
ollama serve & ollama pull qwen2.5-coder:7b
```

```bash
python -m harness.run_judges --judges local --limit-problems 5 --reps 1
```

## Cost

Judging dominates. Approximate figures for the full run as reported in the
paper, at OpenRouter prices:

| Stage | Calls | Model |
|---|---|---|
| Judging, both datasets, 5 judges, 3 reps | ~44,000 | the five judges |
| Instruction control | ~6,600 | the five judges |
| Docstring generation | ~580 | `config.BUILDER` |
| LCB reference solutions | ~470 | `config.BUILDER` |

The whole study came to roughly $50.

Every call is cached on disk under `cache/`, keyed by model, code, spec, schema,
thinking configuration and rep index. Reruns after adding a condition or fixing
the analysis only pay for genuinely new cells. The cache is an optimisation and
is not included here, so a fresh clone rebuilds it.
