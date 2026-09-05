#!/usr/bin/env python3

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def parse_options():
    parser = argparse.ArgumentParser(description="Evaluate selected no-normalization campaign checkpoints on fixed held-out sets.")
    parser.add_argument("--campaign", default="nonorm_n10n20_probe_fixedval")
    parser.add_argument("--output", default="nonorm_n10n20_probe_fixedtest")
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


def main():
    options = parse_options()
    repo = Path(__file__).resolve().parents[1]
    campaign_dir = repo / "results" / "batteries" / options.campaign
    records_path = campaign_dir / "curriculum-checkpoints.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if len(records) != 40:
        raise RuntimeError(f"expected 40 selected checkpoints, found {len(records)}")

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

    for selection in records:
        checkpoint = Path(selection["source_checkpoint"])
        args = json.loads((checkpoint.parent / "args.json").read_text())
        run_name = args["run_name"]
        dataset = test_dataset(repo, args["val_dataset"])
        for decode_strategy, width, eval_batch_size in (("greedy", 0, 1024), ("sample", 1280, 1)):
            artifact = artifacts_dir / f"{run_name}_{decode_strategy}.pkl"
            log = logs_dir / f"{run_name}_{decode_strategy}.log"
            command = [
                str(repo / ".venv" / "bin" / "python"), "eval.py", str(dataset),
                "--model", str(checkpoint), "--decode_strategy", decode_strategy,
                "--width", str(width), "--val_size", "1000", "--eval_batch_size", str(eval_batch_size),
                "--max_calc_batch_size", "10000", "--no_progress_bar", "-o", str(artifact),
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
