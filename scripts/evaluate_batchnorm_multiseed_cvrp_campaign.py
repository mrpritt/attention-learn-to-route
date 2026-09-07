#!/usr/bin/env python3

import argparse
import csv
import json
import shlex
import subprocess
import time
from pathlib import Path


SIZES = (10, 20, 50, 100)
SEEDS = (42, 100, 2024, 999, 1234)
BACKENDS = ("qnn", "classical")


def parse_options():
    parser = argparse.ArgumentParser(description="Evaluate the matched five-seed BatchNorm CVRP campaign on fixed held-out sets.")
    parser.add_argument("--campaign", default="batchnorm_multiseed_cvrp_single_fixedval")
    parser.add_argument("--output", default="batchnorm_multiseed_cvrp_single_fixedtest_s1234")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def test_dataset(repo, validation_dataset):
    name = Path(validation_dataset).name
    suffix = "_val1000_seed4321.pkl"
    if not name.endswith(suffix):
        raise RuntimeError(f"unexpected validation dataset name: {name}")
    path = repo / "data" / "vrp" / (name.removesuffix(suffix) + "_test1000_seed9876.pkl")
    if not path.is_file():
        raise RuntimeError(f"missing held-out dataset: {path}")
    return path


def selected_checkpoints(repo, campaign_dir):
    records_path = campaign_dir / "curriculum-checkpoints.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if len(records) != 32:
        raise RuntimeError(f"expected 32 newly selected checkpoints, found {len(records)}")
    checkpoints = [Path(record["source_checkpoint"]) for record in records]

    adopted_path = campaign_dir / "adopted-seed1234-checkpoints.tsv"
    with adopted_path.open() as source:
        adopted = list(csv.DictReader(source, delimiter="\t"))
    if len(adopted) != 8:
        raise RuntimeError(f"expected 8 adopted seed-1234 checkpoints, found {len(adopted)}")
    checkpoints.extend(repo / row["checkpoint"] for row in adopted)

    selections = []
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise RuntimeError(f"missing selected checkpoint: {checkpoint}")
        args_path = checkpoint.parent / "args.json"
        args = json.loads(args_path.read_text())
        backend = "qnn" if args["encoder_ff_backend"] == "qnn_torch" else "classical"
        key = (args["seed"], backend, args["graph_size"])
        if args["normalization"] != "batch":
            raise RuntimeError(f"selected checkpoint is not BatchNorm: {checkpoint}")
        selections.append((key, checkpoint, args))

    expected = {(seed, backend, size) for seed in SEEDS for backend in BACKENDS for size in SIZES}
    actual = {key for key, _, _ in selections}
    if len(selections) != 40 or actual != expected:
        raise RuntimeError(f"five-seed checkpoint membership mismatch: expected {len(expected)}, found {len(actual)} unique among {len(selections)} records")
    return sorted(selections, key=lambda selection: (SEEDS.index(selection[0][0]), BACKENDS.index(selection[0][1]), SIZES.index(selection[0][2])))


def main():
    options = parse_options()
    repo = Path(__file__).resolve().parents[1]
    campaign_dir = repo / "results" / "batteries" / options.campaign
    selections = selected_checkpoints(repo, campaign_dir)

    output_dir = repo / "results" / options.output
    artifacts_dir = output_dir / "artifacts"
    logs_dir = output_dir / "logs"
    manifest_path = output_dir / "evaluations.jsonl"
    events_path = output_dir / "driver-events.log"
    if not options.dry_run:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

    completed = set()
    if manifest_path.is_file():
        completed = {(row["run_name"], row["decode_strategy"]) for row in map(json.loads, manifest_path.read_text().splitlines())}

    def event(message):
        line = f"=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message} ==="
        print(line, flush=True)
        if not options.dry_run:
            with events_path.open("a") as output:
                output.write(line + "\n")

    for (training_seed, backend, graph_size), checkpoint, args in selections:
        run_name = args["run_name"]
        dataset = test_dataset(repo, args["val_dataset"])
        for decode_strategy, width, eval_batch_size in (("greedy", 0, 1024), ("sample", 1280, 1)):
            artifact = artifacts_dir / f"{run_name}_{decode_strategy}.pkl"
            log = logs_dir / f"{run_name}_{decode_strategy}.log"
            command = [
                str(repo / ".venv" / "bin" / "python"), "eval.py", str(dataset),
                "--model", str(checkpoint), "--decode_strategy", decode_strategy,
                "--width", str(width), "--val_size", "1000", "--eval_batch_size", str(eval_batch_size),
                "--max_calc_batch_size", "10000", "--seed", str(options.seed), "--no_progress_bar", "-o", str(artifact),
            ]
            if options.dry_run:
                print(shlex.join(command))
                continue
            key = (run_name, decode_strategy)
            if key in completed and artifact.is_file():
                event(f"SKIP {run_name} {decode_strategy}")
                continue
            event(f"START {run_name} {decode_strategy}")
            started = time.time()
            with log.open("w") as output:
                result = subprocess.run(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"evaluation failed with status {result.returncode}; see {log}")
            record = {
                "run_name": run_name,
                "training_seed": training_seed,
                "backend": backend,
                "graph_size": graph_size,
                "decoding_seed": options.seed,
                "decode_strategy": decode_strategy,
                "width": width,
                "checkpoint": str(checkpoint.resolve()),
                "dataset": str(dataset.resolve()),
                "artifact": str(artifact.resolve()),
                "log": str(log.resolve()),
                "wall_seconds": time.time() - started,
            }
            with manifest_path.open("a") as output:
                output.write(json.dumps(record) + "\n")
            completed.add(key)
            event(f"END {run_name} {decode_strategy} wall_seconds={record['wall_seconds']:.3f}")
    if not options.dry_run:
        event("HELD-OUT EVALUATION CAMPAIGN COMPLETE")


if __name__ == "__main__":
    main()
