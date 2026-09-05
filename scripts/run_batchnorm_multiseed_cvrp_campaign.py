#!/usr/bin/env python3

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SIZES = (10, 20, 50, 100)
SEEDS = (42, 100, 2024, 999)


@dataclass(frozen=True)
class Chain:
    backend: str
    seed: int

    @property
    def name(self):
        suffix = "classical_b4" if self.backend == "classical" else "qnn_q4l2"
        return f"cvrp_single_{suffix}"


def parse_options():
    parser = argparse.ArgumentParser(description="Run matched BatchNorm CVRP single-compression curricula for four additional seeds.")
    parser.add_argument("--campaign", default="batchnorm_multiseed_cvrp_single_fixedval")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validation_dataset(repo, size):
    path = repo / "data" / "vrp" / f"vrp{size}_frontier_val1000_seed4321.pkl"
    if not path.is_file():
        raise RuntimeError(f"missing validation dataset: {path}")
    return path


def stage_command(repo, campaign, chain, size, load_path=None):
    name = f"{campaign}_{chain.name}_n{size}_s{chain.seed}"
    command = [
        str(repo / ".venv/bin/python"), "run.py", "--problem", "cvrp", "--graph_size", str(size),
        "--batch_size", "128", "--epoch_size", "32768", "--val_size", "1000", "--val_dataset", str(validation_dataset(repo, size)),
        "--n_epochs", "128", "--seed", str(chain.seed), "--baseline", "rollout", "--eval_batch_size", "1024",
        "--normalization", "batch", "--embedding_dim", "128", "--hidden_dim", "128", "--n_encode_layers", "3",
        "--lr_model", "0.0001", "--lr_decay", "1.0", "--max_grad_norm", "1.0",
        "--encoder_ff_hidden", "4", "--encoder_ff_activation", "relu",
        "--qnn_qubits", "4", "--qnn_layers", "2", "--qnn_ansatz", "pce", "--qnn_device", "auto",
        "--qnn_diff_method", "auto", "--qnn_rotation", "RXRYRZ", "--qnn_topology", "brickwall",
        "--encoder_mha_out_backend", "classical", "--encoder_mha_out_layers", "0", "--encoder_mha_out_bottleneck_dim", "4",
        "--checkpoint_epochs", "1", "--no_progress_bar", "--run_name", name,
    ]
    if chain.backend == "qnn":
        command.extend(("--encoder_ff_backend", "qnn_torch", "--encoder_ff_qnn_layers", "3"))
    else:
        command.extend(("--encoder_ff_backend", "classical", "--encoder_ff_qnn_layers", "0"))
    if load_path is not None:
        command.extend(("--load_path", str(load_path)))
    return name, command


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


def validate_args(args_path, chain, size, repo):
    args = json.loads(args_path.read_text())
    expected = {
        "graph_size": size,
        "seed": chain.seed,
        "normalization": "batch",
        "embedding_dim": 128,
        "hidden_dim": 128,
        "n_encode_layers": 3,
        "encoder_ff_hidden": 4,
        "encoder_ff_activation": "relu",
        "encoder_ff_backend": "qnn_torch" if chain.backend == "qnn" else "classical",
        "encoder_ff_qnn_layers": 3 if chain.backend == "qnn" else 0,
        "encoder_mha_out_backend": "classical",
        "encoder_mha_out_layers": 0,
        "qnn_qubits": 4,
        "qnn_layers": 2,
        "n_epochs": 128,
        "epoch_size": 32768,
        "batch_size": 128,
        "val_size": 1000,
    }
    mismatches = {key: (args.get(key), value) for key, value in expected.items() if args.get(key) != value}
    expected_val = validation_dataset(repo, size).resolve()
    if Path(args["val_dataset"]).resolve() != expected_val:
        mismatches["val_dataset"] = (args["val_dataset"], str(expected_val))
    if mismatches:
        raise RuntimeError(f"configuration mismatch in {args_path}: {mismatches}")


def launch_stage(repo, name, command, log_path, events):
    outputs = repo / "outputs" / f"cvrp_{command[command.index('--graph_size') + 1]}"
    before = set(outputs.glob(f"{name}_*"))
    events(f"START {name}")
    with log_path.open("w") as output:
        result = subprocess.run(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"stage {name} failed with status {result.returncode}; see {log_path}")
    created = set(outputs.glob(f"{name}_*")) - before
    if len(created) != 1:
        raise RuntimeError(f"expected one output directory for {name}, found {sorted(created)}")
    run_dir = created.pop()
    events(f"END {name} run_dir={run_dir}")
    return run_dir


def select_transition(run_dir, log_path, transition_dir, records_path, name, preserve_epoch_zero=False):
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
        "normalization": "batch",
        "note": "Rollout baseline state removed; model and optimizer state retained for target-size continuation.",
    }
    with records_path.open("a") as output:
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


def main():
    options = parse_options()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    campaign_dir = repo / "results" / "batteries" / options.campaign
    chains = [Chain(backend, seed) for seed in SEEDS for backend in ("qnn", "classical")]
    schedule = [(chain, size) for chain in chains for size in SIZES]
    if len(schedule) != 32:
        raise AssertionError(f"campaign definition contains {len(schedule)} stages, expected 32")
    if options.dry_run:
        for index, (chain, size) in enumerate(schedule, 1):
            name, command = stage_command(repo, options.campaign, chain, size, Path("PARENT.pt") if size != 10 else None)
            print(f"{index:02d}/32 {name}")
            print("  " + shlex.join(command))
        return

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

    (campaign_dir / "schedule.json").write_text(json.dumps([chain.__dict__ | {"sizes": list(SIZES)} for chain in chains], indent=2) + "\n")
    qnn_epoch_zero = None
    qnn_log = None
    for chain in chains:
        parent = None
        for size in SIZES:
            name, command = stage_command(repo, options.campaign, chain, size, parent)
            log_path = campaign_dir / f"{name}.log"
            run_dir = launch_stage(repo, name, command, log_path, events)
            validate_args(run_dir / "args.json", chain, size, repo)
            preserve = chain.seed == 42 and size == 10
            if preserve and chain.backend == "qnn":
                qnn_epoch_zero = run_dir / "epoch-0.pt"
                qnn_log = log_path
                with (campaign_dir / "qnn-dispatch-preflight.txt").open("w") as output:
                    subprocess.run([str(repo / ".venv/bin/python"), "scripts/check_qnn_dispatch.py", str(qnn_epoch_zero), "--expected-normalization", "batch"], cwd=repo, check=True, stdout=output)
            if preserve and chain.backend == "classical":
                if qnn_epoch_zero is None:
                    raise RuntimeError("missing seed-42 QNN preflight checkpoint")
                with (campaign_dir / "qnn-dispatch-preflight-with-classical.txt").open("w") as output:
                    subprocess.run([str(repo / ".venv/bin/python"), "scripts/check_qnn_dispatch.py", str(qnn_epoch_zero), "--classical-checkpoint", str(run_dir / "epoch-0.pt"), "--expected-normalization", "batch"], cwd=repo, check=True, stdout=output)
                if validation_costs(qnn_log) == validation_costs(log_path):
                    raise RuntimeError("seed-42 QNN and classical validation trajectories are bit-for-bit identical")
                events("PASS QNN/classical dispatch and trajectory comparison")
            parent, record, removed = select_transition(run_dir, log_path, transitions, records_path, name, preserve)
            events(f"SELECT {name} epoch={record['best_epoch']} cost={record['best_fixed_validation_cost']} pruned={removed}")
    events("TRAINING CAMPAIGN COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
