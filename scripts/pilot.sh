#!/usr/bin/env bash
# Pilot: one model, 4 frames, 10 items from A + 10 from B, 2 paraphrases, $2 budget.
# Dry-run first; the real run refuses to start if projected spend exceeds the budget.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=pilot
ARGS=(--run "$RUN" --models pilot --frames F0,F1,F2,F3 --paraphrases 2 --banks A,B --items-per-bank 10 --budget 2)

python -m cracow.runner "${ARGS[@]}" --dry-run
python -m cracow.runner "${ARGS[@]}"
python -m cracow.grader --run "$RUN" --budget 2
python -m cracow.analyze --run "$RUN"

python -m cracow.analyze --run "$RUN" --cost-only
CSV="results/$RUN/${RUN}_graded.csv"
echo "graded CSV: $(pwd)/$CSV"
if [[ -z "${NO_OPEN:-}" ]] && command -v open >/dev/null; then open "$CSV"; fi

echo; echo "== projected cost of scripts/run1.sh (dry run, no calls) =="
python -m cracow.runner --run run1 --models run1 --frames F0,F1,F2,F3 --paraphrases 5 --banks A,B --budget 200 --dry-run
