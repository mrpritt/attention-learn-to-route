#!/usr/bin/env python

import argparse
import csv
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.functions import load_model
from problems import CVRP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export last encoder FF hidden activations for a single CVRP instance"
    )
    parser.add_argument("--model", required=True, help="Model directory or checkpoint file")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--dataset",
        default="data/vrp/vrp20_greedy_sampling_eval1000_seed1234.pkl",
        help="Dataset pickle to draw the instance from"
    )
    parser.add_argument(
        "--instance-index",
        type=int,
        default=0,
        help="Index inside the dataset"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Checkpoint epoch to load; default is latest in model directory"
    )
    return parser.parse_args()


def get_last_ff_hidden_module(model):
    last_layer = model.embedder.layers[-1]
    ff_module = last_layer[2].module
    if not isinstance(ff_module, torch.nn.Sequential):
        raise ValueError("Last encoder FF block is not classical; no hidden bottleneck to export")
    if len(ff_module) < 2 or not isinstance(ff_module[1], torch.nn.ReLU):
        raise ValueError("Expected classical FF block with ReLU hidden activation")
    return ff_module[1]


def make_single_instance(dataset_path, instance_index):
    dataset = CVRP.make_dataset(filename=dataset_path, num_samples=1, offset=instance_index)
    instance = dataset[0]
    return {k: v.unsqueeze(0) for k, v in instance.items()}


def export_rows(instance, hidden, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hidden = hidden.squeeze(0).detach().cpu()
    depot = instance["depot"].squeeze(0).detach().cpu()
    loc = instance["loc"].squeeze(0).detach().cpu()
    demand = instance["demand"].squeeze(0).detach().cpu()

    headers = ["nodeid", "is_depot", "x", "y", "demand"] + [f"h_{i + 1}" for i in range(hidden.size(-1))]

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        writer.writerow([0, True, float(depot[0]), float(depot[1]), 0.0, *map(float, hidden[0].tolist())])
        for idx, (xy, h) in enumerate(zip(loc, hidden[1:]), start=1):
            writer.writerow([idx, False, float(xy[0]), float(xy[1]), float(demand[idx - 1]), *map(float, h.tolist())])


def main():
    args = parse_args()

    model, model_args = load_model(args.model, epoch=args.epoch)
    if model_args["problem"] != "cvrp":
        raise ValueError(f"Only CVRP is supported right now, got {model_args['problem']}")

    instance = make_single_instance(args.dataset, args.instance_index)

    captured = {}

    def save_hidden(_module, _inputs, output):
        captured["hidden"] = output

    hook = get_last_ff_hidden_module(model).register_forward_hook(save_hidden)
    try:
        with torch.no_grad():
            _ = model.embedder(model._init_embed(instance))
    finally:
        hook.remove()

    if "hidden" not in captured:
        raise RuntimeError("Failed to capture encoder FF hidden activations")

    export_rows(instance, captured["hidden"], args.output)


if __name__ == "__main__":
    main()
