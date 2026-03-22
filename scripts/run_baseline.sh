#!/usr/bin/env bash
# Run Kool et al. CVRP classical baseline for given graph sizes.
# Usage: ./run_baseline.sh [n1 n2 ...]
# Default sizes: 20 50 100
#
# Logs and checkpoints go to results/<run-id>/
# where <run-id> = cvrp<n>_classical_<timestamp>

set -euo pipefail

SIZES=${@:-20 50 100}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for N in $SIZES; do
    RUN_ID="cvrp${N}_classical_$(date +%Y%m%dT%H%M%S)"
    RESULTS_DIR="results/${RUN_ID}"
    mkdir -p "$RESULTS_DIR"
    LOG="$RESULTS_DIR/run.log"

    echo "=== Starting n=${N}, run=${RUN_ID} ===" | tee "$LOG"
    echo "Log: $LOG"

    python run.py \
        --problem cvrp \
        --graph_size "$N" \
        --baseline rollout \
        --batch_size 128 \
        --epoch_size 12800 \
        --val_size 1000 \
        --n_epochs 100 \
        --output_dir "$RESULTS_DIR" \
        --run_name "cvrp${N}_classical" \
        --no_tensorboard \
        --project_fixed_context_backend classical \
        2>&1 | tee -a "$LOG"

    echo "=== Finished n=${N} ===" | tee -a "$LOG"
done
