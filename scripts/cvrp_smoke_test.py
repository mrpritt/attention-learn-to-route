import argparse
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets.attention_model import AttentionModel
from problems import CVRP


def build_model(backend):
    return AttentionModel(
        128,
        128,
        CVRP,
        n_encode_layers=3,
        project_fixed_context_backend=backend,
        project_fixed_context_qnn_config={
            "ansatz_name": "pce",
            "n_qubits": 8,
            "n_layers": 4,
            "rotation": "RXRYRZ",
            "topology": "brickwall",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["classical", "qnn"], default="classical")
    parser.add_argument("--graph_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backend).to(device)
    model.train()
    model.set_decode_type("sampling")

    dataset = CVRP.make_dataset(size=args.graph_size, num_samples=args.batch_size)
    batch = {
        key: torch.stack([dataset[i][key] for i in range(args.batch_size)]).to(device)
        for key in ("loc", "demand", "depot")
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start = time.time()
    cost, ll = model(batch)
    loss = (cost * ll).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    elapsed = time.time() - start

    print(
        {
            "backend": args.backend,
            "device": str(device),
            "graph_size": args.graph_size,
            "batch_size": args.batch_size,
            "cost_shape": tuple(cost.shape),
            "ll_shape": tuple(ll.shape),
            "loss": float(loss.detach().cpu()),
            "step_s": elapsed,
        }
    )


if __name__ == "__main__":
    main()
