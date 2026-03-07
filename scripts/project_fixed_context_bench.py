import os
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets.attention_model import AttentionModel
from utils import load_problem


def build_model(backend):
    return AttentionModel(
        128,
        128,
        load_problem("cvrp"),
        n_encode_layers=3,
        project_fixed_context_backend=backend,
        project_fixed_context_qnn_config={
            "ansatz_name": "pce",
            "n_qubits": 8,
            "n_layers": 4,
            "rotation": "RXRYRZ",
            "topology": "brickwall",
        },
    ).eval()


def time_layer(layer, x, warmup=1, repeats=3):
    with torch.no_grad():
        for _ in range(warmup):
            layer(x)
        start = time.time()
        for _ in range(repeats):
            layer(x)
        return (time.time() - start) / repeats


def main():
    batch_size = int(os.getenv("QGAT_BENCH_BATCH", "32"))
    x = torch.randn(batch_size, 128)

    classical = build_model("classical")
    qnn = build_model("qnn")

    classical_s = time_layer(classical.project_fixed_context, x)
    qnn_s = time_layer(qnn.project_fixed_context, x)

    print(
        {
            "batch_size": batch_size,
            "classical_s": classical_s,
            "qnn_s": qnn_s,
            "slowdown": qnn_s / max(classical_s, 1e-12),
        }
    )


if __name__ == "__main__":
    main()
