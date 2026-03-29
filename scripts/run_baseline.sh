#!/usr/bin/env bash
# Run Kool et al. CVRP classical baseline for given graph sizes.
# Usage: ./run_baseline.sh [classical|qnn] [n1 n2 ...]
# Environment overrides for qnn runs:
#   QNN_QUBITS   default: 8
#   QNN_LAYERS   default: 4
# General environment overrides:
#   N_EPOCHS     default: 100
#   STEP_CONTEXT_BACKEND default: classical
# Default sizes: 20 50 100
#
# Logs and checkpoints go to results/kool/
# where model checkpoints land under results/kool/<problem>_<graph_size>/<run-name>/
# and stdout/stderr logs are written alongside the corresponding run directory.

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
N_EPOCHS="${N_EPOCHS:-100}"
STEP_CONTEXT_BACKEND="${STEP_CONTEXT_BACKEND:-classical}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="$REPO_ROOT/results/kool"
cd "$REPO_ROOT"

mkdir -p "$RESULTS_ROOT"

for N in $SIZES; do
    RUN_SUFFIX="${BACKEND}"
    if [[ "$STEP_CONTEXT_BACKEND" != "classical" ]]; then
        RUN_SUFFIX="${RUN_SUFFIX}_step${STEP_CONTEXT_BACKEND}"
    fi
    RUN_NAME="cvrp${N}_${RUN_SUFFIX}"
    RUNS_DIR="$RESULTS_ROOT/cvrp_${N}"
    mkdir -p "$RUNS_DIR"
    TMP_LOG="$(mktemp "$RESULTS_ROOT/${RUN_NAME}_XXXXXX.log")"

    echo "=== Starting n=${N}, run=${RUN_NAME} ===" | tee "$TMP_LOG"
    echo "Temporary log: $TMP_LOG" | tee -a "$TMP_LOG"
    if [[ "$BACKEND" == "qnn" ]]; then
        echo "QNN config: qubits=${QNN_QUBITS}, layers=${QNN_LAYERS}" | tee -a "$TMP_LOG"
    fi
    echo "Step-context backend: ${STEP_CONTEXT_BACKEND}" | tee -a "$TMP_LOG"

    python "$REPO_ROOT/run.py" \
        --problem cvrp \
        --graph_size "$N" \
        --baseline rollout \
        --batch_size 128 \
        --epoch_size 12800 \
        --val_size 1000 \
        --n_epochs "$N_EPOCHS" \
        --output_dir "$RESULTS_ROOT" \
        --run_name "$RUN_NAME" \
        --no_tensorboard \
        --project_fixed_context_backend "$BACKEND" \
        --project_step_context_backend "$STEP_CONTEXT_BACKEND" \
        $(if [[ "$BACKEND" == "qnn" ]]; then
            printf '%s ' \
                --qnn_qubits "$QNN_QUBITS" \
                --qnn_layers "$QNN_LAYERS" \
                --qnn_rotation RXRYRZ \
                --qnn_topology brickwall
        fi) \
        2>&1 | tee -a "$TMP_LOG"

    RUN_DIR="$(find "$RUNS_DIR" -maxdepth 1 -mindepth 1 -type d -name "${RUN_NAME}_*" | sort | tail -n 1)"
    if [[ -n "$RUN_DIR" ]]; then
        FINAL_LOG="${RUN_DIR}.log"
        mv "$TMP_LOG" "$FINAL_LOG"
        echo "Log: $FINAL_LOG"
    else
        echo "Could not determine run directory under $RUNS_DIR; leaving log at $TMP_LOG" >&2
    fi

    echo "=== Finished n=${N} ===" | tee -a "${FINAL_LOG:-$TMP_LOG}"
done
