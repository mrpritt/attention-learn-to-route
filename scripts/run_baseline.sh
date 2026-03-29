#!/usr/bin/env bash
# Run Kool et al. CVRP classical baseline for given graph sizes.
# Usage: ./run_baseline.sh [classical|qnn] [n1 n2 ...]
# Environment overrides for qnn runs:
#   QNN_QUBITS   default: 8
#   QNN_LAYERS   default: 4
# Default sizes: 20 50 100
#
# Logs and checkpoints go to results/kool/
# where model checkpoints land under results/kool/<problem>_<graph_size>/<run-name>/

set -euo pipefail

BACKEND="${1:-classical}"
if [[ "$BACKEND" != "classical" && "$BACKEND" != "qnn" ]]; then
    echo "Usage: $0 [classical|qnn] [n1 n2 ...]" >&2
    exit 1
fi

shift || true
SIZES=${@:-20 50 100}
QNN_QUBITS="${QNN_QUBITS:-8}"
QNN_LAYERS="${QNN_LAYERS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)/results/kool"
cd "$SCRIPT_DIR"

mkdir -p "$RESULTS_ROOT"

for N in $SIZES; do
    RUN_ID="cvrp${N}_${BACKEND}_$(date +%Y%m%dT%H%M%S)"
    LOG="$RESULTS_ROOT/${RUN_ID}.log"

    echo "=== Starting n=${N}, run=${RUN_ID} ===" | tee "$LOG"
    echo "Log: $LOG"
    if [[ "$BACKEND" == "qnn" ]]; then
        echo "QNN config: qubits=${QNN_QUBITS}, layers=${QNN_LAYERS}" | tee -a "$LOG"
    fi

    python run.py \
        --problem cvrp \
        --graph_size "$N" \
        --baseline rollout \
        --batch_size 128 \
        --epoch_size 12800 \
        --val_size 1000 \
        --n_epochs 100 \
        --output_dir "$RESULTS_ROOT" \
        --run_name "cvrp${N}_${BACKEND}" \
        --no_tensorboard \
        --project_fixed_context_backend "$BACKEND" \
        $(if [[ "$BACKEND" == "qnn" ]]; then
            printf '%s ' \
                --qnn_qubits "$QNN_QUBITS" \
                --qnn_layers "$QNN_LAYERS" \
                --qnn_rotation RXRYRZ \
                --qnn_topology brickwall
        fi) \
        2>&1 | tee -a "$LOG"

    echo "=== Finished n=${N} ===" | tee -a "$LOG"
done
