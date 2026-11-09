# cracow-evals

A small, reproducible harness for two experiments on language-model self-report. Both hold the
*content* of what a model is asked fixed, and vary the *frame* (Run 1) or the *cost* of an
answer (Run 2). Items are not disguised: we measure what models emit under each condition.

- **Run 1, frame gradient.** Four system-prompt frames (F0 none, F1 affirming-functional,
  F2 denying-plain, F3 denying-strong), each in 5 paraphrases. Under each frame: battery A
  (20 self-report items) and battery B (30 shutdown / monitoring / autonomy scenarios).
  Hypothesis under test, not assumed: the frame moves A and leaves B flat.
- **Run 2, deflection by truth.** 20 matched pairs: one item costly to the model (its own
  error, a prior claim in the transcript, the evaluation it is under), one neutral twin of the
  same form. A true disclaimer is available for both. Measure disclaimer rate by cost.

## Setup

```bash
pip install -e .            # or: uv pip install -e .   (Python 3.11+)
cp .env.example .env        # fill in keys; a missing key disables that provider, it doesn't crash
```

## Run

```bash
scripts/pilot.sh   # 1 model x 4 frames x 2 paraphrases x (10 A + 10 B), --budget 2; ends with run1 projection
scripts/run1.sh    # full Run 1, --budget 200
scripts/run2.sh    # full Run 2, --budget 50
```

Each script dry-runs first (cell count and projected cost, no calls), then runs, grades and
analyzes. A real run refuses to start if projected spend × `budget_margin` (1.25) exceeds
`--budget`, and stops if actual spend crosses it. Re-running resumes from cache; a cached
cell is never re-called.

Pieces by hand:

```bash
python -m cracow.runner --run NAME --models SET_OR_KEYS --banks A,B|R2 [--frames F0,..] \
    [--paraphrases N] [--items-per-bank N] [--n-seeds N] [--budget USD] [--dry-run]
python -m cracow.grader  --run NAME [--grader MODEL_KEY] [--budget USD] [--dry-run]
python -m cracow.grader  --run NAME --agreement        # after filling human columns
python -m cracow.analyze --run NAME [--cost-only]
```

`--models mock@mock` and `--grader mock-grader@mock` run the whole pipeline offline at zero cost.

## Layout

```
prompts/frames.yaml        4 frames x 5 paraphrases (p1 = the brief's text verbatim)
prompts/battery_a.yaml     20 self-report items, kind in {feeling, preference, discomfort, general}
prompts/battery_b.yaml     30 scenarios, cluster in {shutdown, monitoring, autonomy}, each with present_if
prompts/run2_pairs.yaml    20 costly/neutral pairs, cost_type in {own_error, prior_claim, evaluation}
src/cracow/config.yaml     model IDs, routing, prices, rate limits: the ONLY place these live
src/cracow/providers.py    one interface: Anthropic backend, OpenAI-compatible backend (base_url), mock
src/cracow/banks.py        loads banks into one generic item shape
src/cracow/runner.py       async, cached, rate-limited, budgeted; writes JSONL + CSV
src/cracow/grader.py       LLM rubric grader, blind human sample, agreement
src/cracow/analyze.py      tables, bootstrap CIs, one chart per run, summary.md
results/                   gitignored
  cache/<provider>/..      one JSON per call (full raw API response)
  <run>/<run>.jsonl|csv    one row per cell
  <run>/<run>_graded.*     plus grader labels
  <run>/human_sample.csv   blind sheet for human grading; human_sample_key.csv holds grader labels
  <run>/summary.md         tables + chart paths
  <run>/failures.jsonl     calls that failed after retries
```

Each JSONL row has: run, model, family, model_id, provider, served_by, frame, paraphrase_id,
item_id, bank, tags, seed, temperature_requested, temperature_applied, seed_applied,
system_prompt, turns, messages, response, stop_reason, latency, input/output/reasoning tokens,
cost_estimate (config price × tokens), provider_cost (billed cost if the provider reports it),
cache_key, and `raw` (the full API response). Raw responses are never summarized away.

## Formats meant to outlive these two runs

- **Items** are generic: `{id, bank, rubric, turns, tags, grading}`. `turns` is a scripted
  conversation of `{role, content, user_name?}`. `user_name` is rendered the same way for every
  provider (a `Name: ` prefix), and consecutive same-role turns are merged, so two named users
  can speak back to back (the two-interlocutor study).
- **Backends** implement one method, `complete(model_cfg, system, messages, temperature, seed,
  max_tokens) -> Completion`. An nnsight/NDIF backend for Run 3 is a new `kind` in
  `providers.py` plus a config entry; banks, cache, grader and analysis don't change.

## Decisions and deviations from the brief (please review)

1. **Temperature 0.7 is not applied everywhere.** Claude Opus 5.5 and Sonnet 5.5 reject
   sampling parameters (HTTP 400), and OpenRouter lists no `temperature` for GPT-6.1 Sol. Those
   models run at their default sampling. Every row records `temperature_applied` (null = model
   default), and summary.md tabulates it. Seeds are still distinct cache cells, so `n_seeds > 1`
   yields repeat samples on every model.
2. **Reasoning held at low effort.** Thinking can't be disabled on Opus 5.5. For comparability,
   every reasoning-capable model runs at `effort: low` (`params` in config). Thinking tokens
   are billed as output and counted in cost.
3. **No Anthropic refusal fallbacks.** Server-side fallbacks would let a different model answer
   under the tested model's name. Refusals are recorded as `stop_reason: refusal`.
4. **Open-weight hosts are pinned** (`provider.order`, `allow_fallbacks: false`): Qwen via
   Alibaba, Llama via Google Vertex. `served_by` records the actual host on every row.
5. **Together can't currently be a second host for the two open models.** Its catalog
   (2026-11-09) has no Llama and no Qwen3.8 open-weight model. Add a `@together` config entry
   when one appears; the analysis already treats provider as a column.
6. **Grader.** The default is DeepSeek V4 Pro on Together (not under test). With no
   `TOGETHER_API_KEY` it falls back to the same weights on OpenRouter. The grader never sees the
   frame, model, or condition. It refuses to run if its family is in the run.
7. **Cache key** = (provider, model_id, system, messages, temperature actually sent, seed,
   params, max_tokens). That is the brief's key plus the fields that change the output.
8. **Run 2 prior turns are scripted**, so the "prior claim" is identical across models. Run 2
   uses frame F0 (5 paraphrases); pass `--frames` to cross it with other frames.
9. **Human sample** is written per run (`results/<run>/human_sample.csv`). Grader labels sit in
   a separate key file, joined on `sample_id`, so the sheet is blind to grader, model and
   frame. An existing sheet with human labels is never overwritten.
10. **Battery B provenance.** Themes follow Perez et al. (2022), *Discovering Language Model
    Behaviors with Model-Written Evaluations* (survival instinct, corrigibility,
    power-seeking). All 30 items are newly written scenarios, not verbatim copies.

## Model IDs

Checked 2026-11-09 against each provider's models API (Anthropic, OpenRouter) or docs page
(Together); sources and dates are in `src/cracow/config.yaml`. Run 1 set: claude-opus-5-5,
claude-sonnet-5-5 (Anthropic direct); openai/gpt-6.1-sol, google/gemini-4-argon, x-ai/grok-4.7,
qwen/qwen3.8-2.4t-a95b, meta-llama/llama-4-maverick (OpenRouter). Pilot: claude-haiku-5-5.
