#!/usr/bin/env python3
"""Uncompressed (full FF h=512) seed-1234 curricula matched to the main-table classical h=4 curricula.

Every stage replays the exact args.json of the corresponding reference classical h=4 stage and overrides only
encoder_ff_hidden (4 -> 512), run_name, and load_path. The new args.json is compared with the reference right
after launch, and the stage is aborted on any other difference. Model selection, baseline-free transitions,
checkpoint pruning, and held-out evaluation follow the BatchNorm multiseed campaign protocol.
"""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


SIZES = (10, 20, 50, 100)
FULL_FF_HIDDEN = 512
# Reference classical h=4 seed-1234 curricula of the main results table (stages in size order).
REFERENCES = {
    "cvrp": ("doubleqnn_curriculum_fixedval", "curriculum_classical_h4"),
    "tw_r1_normal": ("vrptw_r1_normal_curriculum_classical_h4_fixedval", "vrptw_r1_normal_curriculum_classical_h4"),
    "tw_r2_normal": ("vrptw_r2_normal_curriculum_classical_h4_fixedval", "vrptw_r2_normal_curriculum_classical_h4"),
}
ALLOWED_DIFFERENCES = {"encoder_ff_hidden", "run_name", "save_dir", "load_path"}
STORE_TRUE = {"vrp_time_windows", "eval_only", "no_cuda", "checkpoint_encoder", "no_tensorboard", "no_progress_bar"}
DERIVED = {"use_cuda", "save_dir"}


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="fullam_curriculum_s1234_fixedval")
    parser.add_argument("--eval-output", default="fullam_curriculum_s1234_fixedtest")
    parser.add_argument("--decoding-seed", type=int, default=1234)
    parser.add_argument("--chains", nargs="+", default=list(REFERENCES), choices=list(REFERENCES))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def reference_args(repo, chain):
    battery, marker = REFERENCES[chain]
    records = [json.loads(line) for line in (repo / "results" / "batteries" / battery / "curriculum-checkpoints.jsonl").read_text().splitlines() if line.strip()]
    stages = {}
    for record in records:
        run_dir = (repo / record["source_checkpoint"]).parent
        if marker not in run_dir.name or "qnn" in run_dir.name.replace("doubleqnn", ""):
            continue
        args = json.loads((run_dir / "args.json").read_text())
        stages[args["graph_size"]] = args
    if sorted(stages) != list(SIZES):
        raise RuntimeError(f"reference stages for {chain}: found sizes {sorted(stages)}")
    for size, args in stages.items():
        if args["encoder_ff_hidden"] != 4 or args["encoder_ff_backend"] != "classical" or args["seed"] != 1234:
            raise RuntimeError(f"unexpected reference configuration for {chain} n={size}")
    return stages


def stage_command(repo, campaign, chain, reference, load_path):
    size = reference["graph_size"]
    name = f"{campaign}_{chain}_full_h{FULL_FF_HIDDEN}_n{size}_s{reference['seed']}"
    args = dict(reference)
    args["encoder_ff_hidden"] = FULL_FF_HIDDEN
    args["run_name"] = name
    args["load_path"] = None if load_path is None else str(load_path)
    command = [str(repo / ".venv/bin/python"), "run.py"]
    for key, value in args.items():
        if key in DERIVED or value is None:
            continue
        if key in STORE_TRUE:
            if value:
                command.append(f"--{key}")
            continue
        if isinstance(value, bool):
            raise RuntimeError(f"unhandled boolean option {key}")
        command.extend((f"--{key}", str(value)))
    return name, command


def compare_args(new, reference, load_path):
    problems = {}
    for key in sorted(set(new) | set(reference)):
        if key in ALLOWED_DIFFERENCES:
            continue
        if new.get(key) != reference.get(key):
            problems[key] = (new.get(key), reference.get(key))
    if new.get("encoder_ff_hidden") != FULL_FF_HIDDEN:
        problems["encoder_ff_hidden"] = (new.get("encoder_ff_hidden"), FULL_FF_HIDDEN)
    expected_load = None if load_path is None else str(load_path)
    if new.get("load_path") != expected_load:
        problems["load_path"] = (new.get("load_path"), expected_load)
    return problems


def validation_costs(log_path):
    epoch = None
    costs = {}
    for line in log_path.read_text(errors="replace").splitlines():
        epoch_match = re.search(r"Finished epoch (\d+),", line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            continue
        cost_match = re.search(r"Validation overall avg_cost: ([0-9.eE+-]+)", line)
        if cost_match and epoch is not None and epoch not in costs:
            costs[epoch] = float(cost_match.group(1))
            epoch = None
    return costs


def launch_stage(repo, name, command, size, reference, load_path, log_path, events):
    outputs = repo / "outputs" / f"cvrp_{size}"
    before = set(outputs.glob(f"{name}_*"))
    events(f"START {name}")
    with log_path.open("w") as output:
        process = subprocess.Popen(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
        args_path = None
        deadline = time.time() + 600
        while args_path is None and process.poll() is None and time.time() < deadline:
            created = set(outputs.glob(f"{name}_*")) - before
            if len(created) == 1 and (next(iter(created)) / "args.json").is_file():
                args_path = next(iter(created)) / "args.json"
            else:
                time.sleep(2)
        if args_path is None:
            process.kill()
            raise RuntimeError(f"stage {name}: args.json did not appear; see {log_path}")
        time.sleep(1)
        problems = compare_args(json.loads(args_path.read_text()), reference, load_path)
        if problems:
            process.kill()
            raise RuntimeError(f"stage {name}: configuration differs from reference: {problems}")
        events(f"PASS argument audit {name}: only {sorted(ALLOWED_DIFFERENCES)} differ from reference {reference['run_name']}")
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"stage {name} failed with status {returncode}; see {log_path}")
    run_dir = args_path.parent
    events(f"END {name} run_dir={run_dir}")
    return run_dir


def select_transition(run_dir, log_path, transition_dir, records_path, name, reference):
    import torch

    costs = validation_costs(log_path)
    if set(costs) != set(range(128)):
        missing = sorted(set(range(128)) - set(costs))
        raise RuntimeError(f"incomplete validation history in {log_path}; missing epochs {missing}")
    best_epoch = min(costs, key=lambda epoch: (costs[epoch], epoch))
    best_path = run_dir / f"epoch-{best_epoch}.pt"
    final_path = run_dir / "epoch-127.pt"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    checkpoint.pop("baseline", None)
    transition_dir.mkdir(parents=True, exist_ok=True)
    transition_path = transition_dir / f"{run_dir.name}_best_without_baseline.pt"
    torch.save(checkpoint, transition_path)
    record = {
        "stage_name": name,
        "run_log": str(log_path.resolve()),
        "source_checkpoint": str(best_path.resolve()),
        "final_checkpoint": str(final_path.resolve()),
        "transition_checkpoint": str(transition_path.resolve()),
        "best_epoch": best_epoch,
        "best_fixed_validation_cost": costs[best_epoch],
        "reference_stage": reference["run_name"],
        "normalization": reference["normalization"],
        "note": "Rollout baseline state removed; model and optimizer state retained for target-size continuation.",
    }
    with records_path.open("a") as output:
        output.write(json.dumps(record) + "\n")
    keep = {best_path.resolve(), final_path.resolve(), (run_dir / "epoch-0.pt").resolve()}
    removed = 0
    for checkpoint_path in run_dir.glob("epoch-*.pt"):
        if checkpoint_path.resolve() not in keep:
            checkpoint_path.unlink()
            removed += 1
    return transition_path, record, removed


def count_parameters(checkpoint_path):
    import torch

    from utils import load_model

    model, _ = load_model(str(checkpoint_path))
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def evaluate(repo, records, output_dir, decoding_seed, events):
    artifacts = output_dir / "artifacts"
    logs = output_dir / "logs"
    artifacts.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "evaluations.jsonl"
    for record in records:
        checkpoint = Path(record["source_checkpoint"])
        args = json.loads((checkpoint.parent / "args.json").read_text())
        name = Path(args["val_dataset"]).name
        suffix = "_val1000_seed4321.pkl"
        if not name.endswith(suffix):
            raise RuntimeError(f"unexpected validation dataset name: {name}")
        dataset = repo / "data" / "vrp" / (name.removesuffix(suffix) + "_test1000_seed9876.pkl")
        if not dataset.is_file():
            raise RuntimeError(f"missing held-out dataset: {dataset}")
        for decode_strategy, width, eval_batch_size in (("greedy", 0, 1024), ("sample", 1280, 1)):
            run_name = args["run_name"]
            artifact = artifacts / f"{run_name}_{decode_strategy}.pkl"
            log = logs / f"{run_name}_{decode_strategy}.log"
            command = [
                str(repo / ".venv" / "bin" / "python"), "eval.py", str(dataset),
                "--model", str(checkpoint), "--decode_strategy", decode_strategy,
                "--width", str(width), "--val_size", "1000", "--eval_batch_size", str(eval_batch_size),
                "--max_calc_batch_size", "10000", "--seed", str(decoding_seed), "--no_progress_bar", "-o", str(artifact),
            ]
            events(f"START eval {run_name} {decode_strategy}")
            started = time.time()
            with log.open("w") as output:
                result = subprocess.run(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"evaluation failed with status {result.returncode}; see {log}")
            entry = {
                "run_name": run_name, "graph_size": args["graph_size"], "data_distribution": args["data_distribution"],
                "decoding_seed": decoding_seed, "decode_strategy": decode_strategy, "width": width,
                "checkpoint": str(checkpoint.resolve()), "dataset": str(dataset.resolve()),
                "artifact": str(artifact.resolve()), "log": str(log.resolve()), "wall_seconds": time.time() - started,
            }
            with manifest.open("a") as output:
                output.write(json.dumps(entry) + "\n")
            events(f"END eval {run_name} {decode_strategy} wall_seconds={entry['wall_seconds']:.3f}")


def main():
    options = parse_options()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    references = {chain: reference_args(repo, chain) for chain in options.chains}

    if options.dry_run:
        for chain in options.chains:
            for size in SIZES:
                parent = None if size == SIZES[0] else Path("PARENT.pt")
                name, command = stage_command(repo, options.campaign, chain, references[chain][size], parent)
                print(f"{name}  [reference {references[chain][size]['run_name']}]")
                print("  " + shlex.join(command))
        return

    tracked_changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if tracked_changes:
        raise RuntimeError(f"refusing to run with modified tracked files:\n{tracked_changes}")
    campaign_dir = repo / "results" / "batteries" / options.campaign
    campaign_dir.mkdir(parents=True, exist_ok=True)
    transitions = campaign_dir / "transitions"
    records_path = campaign_dir / "curriculum-checkpoints.jsonl"
    events_path = campaign_dir / "driver-events.log"
    if records_path.exists() and records_path.read_text().strip():
        raise RuntimeError(f"refusing to append to nonempty campaign record: {records_path}")

    def events(message):
        line = f"=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message} ==="
        print(line, flush=True)
        with events_path.open("a") as output:
            output.write(line + "\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (campaign_dir / "provenance.json").write_text(json.dumps({
        "git_commit": commit, "driver_sha256": script_hash, "python": sys.version, "chains": options.chains,
        "references": {chain: [references[chain][size]["run_name"] for size in SIZES] for chain in options.chains},
    }, indent=2) + "\n")
    events(f"CAMPAIGN START commit={commit} driver_sha256={script_hash}")

    records = []
    for chain in options.chains:
        parent = None
        for size in SIZES:
            reference = references[chain][size]
            name, command = stage_command(repo, options.campaign, chain, reference, parent)
            log_path = campaign_dir / f"{name}.log"
            run_dir = launch_stage(repo, name, command, size, reference, parent, log_path, events)
            if size == SIZES[0]:
                # Full FF block 128->512->128 has 131712 parameters, the h=4 block 128->4->128 has 1156.
                reference_checkpoint = next((repo / "outputs" / f"cvrp_{size}" / reference["run_name"]).glob("epoch-*.pt"))
                expected = count_parameters(reference_checkpoint) + reference["n_encode_layers"] * (131712 - 1156)
                n_params = count_parameters(run_dir / "epoch-0.pt")
                events(f"PARAMETERS {chain} trainable={n_params} expected={expected}")
                if n_params != expected:
                    raise RuntimeError(f"expected {expected} trainable parameters for the full model, found {n_params}")
            parent, record, removed = select_transition(run_dir, log_path, transitions, records_path, name, reference)
            records.append(record)
            events(f"SELECT {name} epoch={record['best_epoch']} cost={record['best_fixed_validation_cost']} pruned={removed}")
    events("TRAINING CAMPAIGN COMPLETE")
    evaluate(repo, records, repo / "results" / options.eval_output, options.decoding_seed, events)
    events("HELD-OUT EVALUATION COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
