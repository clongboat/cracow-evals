#!/usr/bin/env bash
# Run 2, deflection by truth: run2 model set x neutral frame F0 (5 paraphrases) x 20 pairs x 2 conditions.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=run2
ARGS=(--run "$RUN" --models run2 --frames F0 --paraphrases 5 --banks R2 --budget 50)
python -m cracow.runner "${ARGS[@]}" --dry-run
python -m cracow.runner "${ARGS[@]}"
python -m cracow.grader --run "$RUN" --budget 50
python -m cracow.analyze --run "$RUN"
