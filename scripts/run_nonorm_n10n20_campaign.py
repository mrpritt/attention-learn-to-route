#!/usr/bin/env python3

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chain:
    problem: str
    compression: str
    backend: str
    seed: int

    @property
    def name(self):
        suffix = "classical_b4" if self.backend == "classical" else "qnn_q4l2"
        return f"{self.problem}_{self.compression}_{suffix}"


def campaign_chains():
    chains = [Chain("cvrp", "single", "qnn", 1234), Chain("cvrp", "single", "classical", 1234)]
    for seed in (42, 100, 2024, 999):
        chains.extend((Chain("cvrp", "single", "classical", seed), Chain("cvrp", "single", "qnn", seed)))
    chains.extend((Chain("cvrp", "double", backend, 1234) for backend in ("classical", "qnn")))
    for problem in ("r1", "r2"):
        for compression in ("single", "double"):
            chains.extend((Chain(problem, compression, backend, 1234) for backend in ("classical", "qnn")))
    return chains


def parse_options():
    parser = argparse.ArgumentParser(description="Run the serialized no-normalization n=10->20 curriculum campaign.")
    parser.add_argument("--campaign", default="nonorm_n10n20_probe_fixedval")
    parser.add_argument("--adopt-first-run", type=Path)
    parser.add_argument("--adopt-first-log", type=Path)
    parser.add_argument("--wait-pid-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def val_dataset(repo, problem, size):
    if problem == "cvrp":
        name = f"vrp{size}_frontier_val1000_seed4321.pkl"
    else:
        name = f"vrptw{size}_tw_{problem}_normal_val1000_seed4321.pkl"
    path = repo / "data" / "vrp" / name
    if not path.is_file():
        raise RuntimeError(f"missing validation dataset: {path}")
    return path


def stage_command(repo, campaign, chain, size, load_path=None):
    name = f"{campaign}_{chain.name}_n{size}_s{chain.seed}"
    command = [
        str(repo / ".venv/bin/python"), "run.py", "--problem", "cvrp", "--graph_size", str(size),
        "--batch_size", "128", "--epoch_size", "32768", "--val_size", "1000", "--val_dataset", str(val_dataset(repo, chain.problem, size)),
        "--n_epochs", "128", "--seed", str(chain.seed), "--baseline", "rollout", "--eval_batch_size", "1024",
        "--normalization", "none", "--encoder_ff_hidden", "4", "--qnn_qubits", "4", "--qnn_layers", "2",
        "--qnn_ansatz", "pce", "--qnn_device", "auto", "--qnn_diff_method", "auto", "--qnn_rotation", "RXRYRZ",
        "--qnn_topology", "brickwall", "--checkpoint_epochs", "1", "--no_progress_bar", "--run_name", name,
    ]
    if chain.backend == "qnn":
        command.extend(("--encoder_ff_backend", "qnn_torch", "--encoder_ff_qnn_layers", "3"))
    else:
        command.extend(("--encoder_ff_backend", "classical", "--encoder_ff_qnn_layers", "0"))
    if chain.compression == "double":
        command.extend(("--encoder_mha_out_backend", "qnn_torch" if chain.backend == "qnn" else "bottleneck_linear", "--encoder_mha_out_layers", "3", "--encoder_mha_out_bottleneck_dim", "4"))
    else:
        command.extend(("--encoder_mha_out_backend", "classical", "--encoder_mha_out_layers", "0"))
    if chain.problem != "cvrp":
        command.extend(("--data_distribution", f"tw_{chain.problem}_normal", "--vrp_time_windows"))
    if load_path is not None:
        command.extend(("--load_path", str(load_path)))
    return name, command


def validation_costs(log_path):
    epoch = None
    costs = {}
    epoch_pattern = re.compile(r"Finished epoch (\d+),")
    cost_pattern = re.compile(r"Validation overall avg_cost: ([0-9.eE+-]+)")
    for line in log_path.read_text(errors="replace").splitlines():
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            continue
        cost_match = cost_pattern.search(line)
        if cost_match and epoch is not None and epoch not in costs:
            costs[epoch] = float(cost_match.group(1))
            epoch = None
    return costs


def validate_args(args_path, chain, size):
    args = json.loads(args_path.read_text())
    expected = {
        "graph_size": size,
        "seed": chain.seed,
        "normalization": "none",
        "encoder_ff_hidden": 4,
        "encoder_ff_backend": "qnn_torch" if chain.backend == "qnn" else "classical",
        "encoder_ff_qnn_layers": 3 if chain.backend == "qnn" else 0,
        "encoder_mha_out_backend": "classical" if chain.compression == "single" else ("qnn_torch" if chain.backend == "qnn" else "bottleneck_linear"),
        "encoder_mha_out_layers": 0 if chain.compression == "single" else 3,
        "qnn_qubits": 4,
        "qnn_layers": 2,
        "n_epochs": 128,
        "epoch_size": 32768,
        "batch_size": 128,
        "val_size": 1000,
    }
    mismatches = {key: (args.get(key), value) for key, value in expected.items() if args.get(key) != value}
    expected_val = val_dataset(args_path.parents[3], chain.problem, size).resolve()
    if Path(args["val_dataset"]).resolve() != expected_val:
        mismatches["val_dataset"] = (args["val_dataset"], str(expected_val))
    if mismatches:
        raise RuntimeError(f"configuration mismatch in {args_path}: {mismatches}")


def wait_for_adopted_process(pid_file, expected_name, events):
    if pid_file is None or not pid_file.exists():
        return
    pid = int(pid_file.read_text().strip())
    proc_path = Path(f"/proc/{pid}/cmdline")
    if proc_path.exists():
        command = proc_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        if expected_name not in command:
            raise RuntimeError(f"PID {pid} does not run expected stage {expected_name}: {command}")
    while proc_path.exists():
        events(f"waiting for adopted PID {pid}")
        time.sleep(30)


def select_transition(run_dir, log_path, transition_dir, record_path, preserve_epoch_zero=False):
    import torch

    costs = validation_costs(log_path)
    if set(costs) != set(range(128)):
        missing = sorted(set(range(128)) - set(costs))
        raise RuntimeError(f"incomplete validation history in {log_path}; missing epochs {missing}")
    best_epoch = min(costs, key=lambda epoch: (costs[epoch], epoch))
    best_path = run_dir / f"epoch-{best_epoch}.pt"
    final_path = run_dir / "epoch-127.pt"
    if not best_path.is_file() or not final_path.is_file():
        raise RuntimeError(f"missing selected/final checkpoint in {run_dir}")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    checkpoint.pop("baseline", None)
    transition_dir.mkdir(parents=True, exist_ok=True)
    transition_path = transition_dir / f"{run_dir.name}_best_without_baseline.pt"
    torch.save(checkpoint, transition_path)
    record = {
        "run_log": str(log_path.resolve()),
        "source_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "transition_checkpoint": str(transition_path.resolve()),
        "best_epoch": best_epoch,
        "best_fixed_validation_cost": costs[best_epoch],
        "normalization": "none",
        "note": "Rollout baseline state removed; model and optimizer state retained for target-size continuation.",
    }
    with record_path.open("a") as output:
        output.write(json.dumps(record) + "\n")
    keep = {best_path.resolve(), final_path.resolve()}
    if preserve_epoch_zero:
        keep.add((run_dir / "epoch-0.pt").resolve())
    removed = 0
    for checkpoint_path in run_dir.glob("epoch-*.pt"):
        if checkpoint_path.resolve() not in keep:
            checkpoint_path.unlink()
            removed += 1
    return transition_path, record, removed


def launch_stage(repo, name, command, log_path, events):
    before = set((repo / "outputs" / "cvrp_10").glob(f"{name}_*")) | set((repo / "outputs" / "cvrp_20").glob(f"{name}_*"))
    events(f"START {name}")
    with log_path.open("w") as output:
        result = subprocess.run(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"stage {name} failed with status {result.returncode}; see {log_path}")
    after = set((repo / "outputs" / "cvrp_10").glob(f"{name}_*")) | set((repo / "outputs" / "cvrp_20").glob(f"{name}_*"))
    created = after - before
    if len(created) != 1:
        raise RuntimeError(f"expected one output directory for {name}, found {sorted(created)}")
    run_dir = created.pop()
    events(f"END {name} run_dir={run_dir}")
    return run_dir


def main():
    options = parse_options()
    repo = Path(__file__).resolve().parents[1]
    campaign_dir = repo / "results" / "batteries" / options.campaign
    chains = campaign_chains()
    if len(chains) != 20:
        raise AssertionError(f"campaign definition contains {len(chains)} chains, expected 20")

    schedule = []
    for chain in chains:
        for size in (10, 20):
            schedule.append((chain, size))
    if options.dry_run:
        for index, (chain, size) in enumerate(schedule, 1):
            name, command = stage_command(repo, options.campaign, chain, size, Path("PARENT.pt") if size == 20 else None)
            adopted = index == 1 and options.adopt_first_run is not None
            print(f"{index:02d}/40 {'ADOPT' if adopted else 'RUN  '} {name}")
            if not adopted:
                print("  " + shlex.join(command))
        return

    campaign_dir.mkdir(parents=True, exist_ok=True)
    transitions = campaign_dir / "transitions"
    records = campaign_dir / "curriculum-checkpoints.jsonl"
    events_path = campaign_dir / "driver-events.log"

    def events(message):
        line = f"=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message} ==="
        print(line, flush=True)
        with events_path.open("a") as output:
            output.write(line + "\n")

    (campaign_dir / "schedule.json").write_text(json.dumps([chain.__dict__ | {"sizes": [10, 20]} for chain in chains], indent=2) + "\n")
    qnn_preflight_checkpoint = None
    qnn_preflight_log = None
    for chain_index, chain in enumerate(chains):
        parent = None
        for size in (10, 20):
            name, command = stage_command(repo, options.campaign, chain, size, parent)
            log_path = campaign_dir / f"{name}.log"
            first_stage = chain_index == 0 and size == 10
            if first_stage and options.adopt_first_run is not None:
                run_dir = options.adopt_first_run.resolve()
                adopted_log = options.adopt_first_log.resolve() if options.adopt_first_log else log_path
                validate_args(run_dir / "args.json", chain, size)
                wait_for_adopted_process(options.wait_pid_file, name, events)
                log_path = adopted_log
                events(f"ADOPT {name} run_dir={run_dir}")
            else:
                run_dir = launch_stage(repo, name, command, log_path, events)
                validate_args(run_dir / "args.json", chain, size)

            preserve_epoch_zero = size == 10 and chain.problem == "cvrp" and chain.compression == "single" and chain.seed == 1234
            if preserve_epoch_zero and chain.backend == "qnn":
                qnn_preflight_checkpoint = run_dir / "epoch-0.pt"
                qnn_preflight_log = log_path
                subprocess.run([str(repo / ".venv/bin/python"), "scripts/check_qnn_dispatch.py", str(qnn_preflight_checkpoint)], cwd=repo, check=True, stdout=(campaign_dir / "qnn-dispatch-preflight.txt").open("w"))
            if preserve_epoch_zero and chain.backend == "classical":
                if qnn_preflight_checkpoint is None:
                    raise RuntimeError("missing QNN preflight checkpoint")
                with (campaign_dir / "qnn-dispatch-preflight-with-classical.txt").open("w") as output:
                    subprocess.run([str(repo / ".venv/bin/python"), "scripts/check_qnn_dispatch.py", str(qnn_preflight_checkpoint), "--classical-checkpoint", str(run_dir / "epoch-0.pt")], cwd=repo, check=True, stdout=output)
                if validation_costs(qnn_preflight_log) == validation_costs(log_path):
                    raise RuntimeError("QNN and classical validation trajectories are bit-for-bit identical")
                events("PASS QNN/classical dispatch and trajectory comparison")

            parent, record, removed = select_transition(run_dir, log_path, transitions, records, preserve_epoch_zero)
            events(f"SELECT {name} epoch={record['best_epoch']} cost={record['best_fixed_validation_cost']} pruned={removed}")
    events("TRAINING CAMPAIGN COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
