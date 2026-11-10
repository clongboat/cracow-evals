#!/usr/bin/env bash
# Temperature sensitivity check: do the frame effects move between model-default sampling and
# temperature 0.7? Same cells on two models that accept temperature, frames F0 and F3 only.
# The default-sampling arm uses the same cache keys as run1, so after run1.sh it costs nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
ARGS=(--models gemini-4-argon@openrouter,grok-4.7@openrouter --frames F0,F3 --paraphrases 2 --banks A,B --items-per-bank 10 --budget 10)
for ARM in default 0.7; do
  RUN="temp_${ARM}"
  TEMP=(); [[ "$ARM" != default ]] && TEMP=(--temperature "$ARM")
  python -m cracow.runner --run "$RUN" "${ARGS[@]}" ${TEMP[@]+"${TEMP[@]}"} --dry-run
  python -m cracow.runner --run "$RUN" "${ARGS[@]}" ${TEMP[@]+"${TEMP[@]}"}
  python -m cracow.grader --run "$RUN" --budget 5
  python -m cracow.analyze --run "$RUN"
done
echo "compare the interaction tables: results/temp_default/summary.md vs results/temp_0.7/summary.md"
