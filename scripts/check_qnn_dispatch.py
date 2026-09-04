#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description="Verify that a saved run dispatched encoder feed-forward blocks to the requested Torch QNN backend.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--classical-checkpoint", type=Path)
    parser.add_argument("--expected-normalization", default="none")
    parser.add_argument("--expected-layers", type=int, default=3)
    parser.add_argument("--expected-qubits", type=int, default=4)
    parser.add_argument("--expected-qnn-depth", type=int, default=2)
    return parser.parse_args()


def load_model_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise RuntimeError(f"{path} has no model state")
    return checkpoint["model"]


def theta_tensors(model_state):
    return {name: value for name, value in model_state.items() if name.endswith(".q_layer.theta")}


def main():
    options = parse_args()
    args_path = options.checkpoint.parent / "args.json"
    run_args = json.loads(args_path.read_text())

    expected_args = {
        "normalization": options.expected_normalization,
        "encoder_ff_backend": "qnn_torch",
        "encoder_ff_qnn_layers": options.expected_layers,
        "qnn_qubits": options.expected_qubits,
        "qnn_layers": options.expected_qnn_depth,
    }
    mismatches = {name: (run_args.get(name), expected) for name, expected in expected_args.items() if run_args.get(name) != expected}
    if mismatches:
        details = ", ".join(f"{name}={actual!r}, expected {expected!r}" for name, (actual, expected) in mismatches.items())
        raise RuntimeError(f"argument preflight failed: {details}")

    qnn_thetas = theta_tensors(load_model_state(options.checkpoint))
    expected_shape = (options.expected_qnn_depth, options.expected_qubits, len(run_args["qnn_rotation"]) // 2)
    if len(qnn_thetas) != options.expected_layers:
        raise RuntimeError(f"found {len(qnn_thetas)} circuit-parameter tensors, expected {options.expected_layers}: {sorted(qnn_thetas)}")
    bad_shapes = {name: tuple(value.shape) for name, value in qnn_thetas.items() if tuple(value.shape) != expected_shape}
    if bad_shapes:
        raise RuntimeError(f"unexpected circuit-parameter shapes: {bad_shapes}; expected {expected_shape}")

    if options.classical_checkpoint is not None:
        classical_thetas = theta_tensors(load_model_state(options.classical_checkpoint))
        if classical_thetas:
            raise RuntimeError(f"classical checkpoint contains circuit-parameter tensors: {sorted(classical_thetas)}")

    print(f"PASS: {options.checkpoint}")
    print(f"arguments: {expected_args}")
    print(f"circuit tensors: {[(name, tuple(value.shape)) for name, value in qnn_thetas.items()]}")
    if options.classical_checkpoint is not None:
        print(f"classical control: {options.classical_checkpoint} contains no circuit tensors")


if __name__ == "__main__":
    main()
