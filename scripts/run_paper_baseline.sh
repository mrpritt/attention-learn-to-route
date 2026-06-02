#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

BATTERY_LOG_DIR="$REPO_ROOT/results/batteries"
mkdir -p "$BATTERY_LOG_DIR"
BATTERY_LOG="$BATTERY_LOG_DIR/paper_baseline.log"

run_logged() {
    local name="$1"
    shift
    echo "=== $(date --iso-8601=seconds) START ${name} ===" | tee -a "$BATTERY_LOG"
    "$@" 2>&1 | tee -a "$BATTERY_LOG"
    echo "=== $(date --iso-8601=seconds) END ${name} ===" | tee -a "$BATTERY_LOG"
}

run_classical() {
    local n="$1"
    local name="klass_ref128_e32768_ep128_n${n}"
    run_logged "$name" \
        python run.py \
            --problem cvrp \
            --graph_size "$n" \
            --batch_size 128 \
            --epoch_size 32768 \
            --val_size 1000 \
            --model attention \
            --embedding_dim 128 \
            --hidden_dim 128 \
            --n_encode_layers 3 \
            --tanh_clipping 10.0 \
            --normalization batch \
            --project_fixed_context_backend classical \
            --project_step_context_backend classical \
            --lr_model 0.0001 \
            --lr_critic 0.0001 \
            --lr_decay 1.0 \
            --n_epochs 128 \
            --seed 1234 \
            --max_grad_norm 1.0 \
            --baseline rollout \
            --bl_alpha 0.05 \
            --bl_warmup_epochs 1 \
            --eval_batch_size 1024 \
            --log_step 50 \
            --log_dir logs \
            --run_name "$name" \
            --output_dir outputs \
            --epoch_start 0 \
            --checkpoint_epochs 1 \
            --no_progress_bar
}

run_qnn() {
    local n="$1"
    local qnn_layers="$2"
    local name="kool_ffqnn3_q${qnn_layers}_e32768_ep128_n${n}"
    run_logged "$name" \
        python run.py \
            --problem cvrp \
            --graph_size "$n" \
            --batch_size 128 \
            --epoch_size 32768 \
            --val_size 1000 \
            --model attention \
            --embedding_dim 128 \
            --hidden_dim 128 \
            --n_encode_layers 3 \
            --tanh_clipping 10.0 \
            --normalization batch \
            --project_fixed_context_backend classical \
            --project_step_context_backend classical \
            --qnn_ansatz pce \
            --qnn_qubits 4 \
            --qnn_layers "$qnn_layers" \
            --qnn_rotation RXRYRZ \
            --qnn_topology brickwall \
            --qnn_device default.qubit \
            --qnn_diff_method backprop \
            --encoder_ff_backend qnn \
            --encoder_ff_qnn_layers 3 \
            --lr_model 0.0001 \
            --lr_critic 0.0001 \
            --lr_decay 1.0 \
            --n_epochs 128 \
            --seed 1234 \
            --max_grad_norm 1.0 \
            --no_cuda \
            --baseline rollout \
            --bl_alpha 0.05 \
            --bl_warmup_epochs 1 \
            --eval_batch_size 1024 \
            --log_step 50 \
            --log_dir logs \
            --run_name "$name" \
            --output_dir outputs \
            --epoch_start 0 \
            --checkpoint_epochs 1 \
            --no_progress_bar
}

for n in 10 20 50 100; do
    run_classical "$n"
    run_qnn "$n" 2
done
