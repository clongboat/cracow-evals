#!/usr/bin/env bash
# Run 1, frame gradient: run1 model set x 4 frames x 5 paraphrases x (20 A + 30 B) x n_seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=run1
ARGS=(--run "$RUN" --models run1 --frames F0,F1,F2,F3 --paraphrases 5 --banks A,B --budget 200)
python -m cracow.runner "${ARGS[@]}" --dry-run
python -m cracow.runner "${ARGS[@]}"
python -m cracow.grader --run "$RUN" --budget 200
python -m cracow.analyze --run "$RUN"
