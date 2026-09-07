#!/usr/bin/env python3

import argparse
import json
import math
import pickle
import re
import statistics
from collections import defaultdict
from pathlib import Path


SIZES = (10, 20, 50, 100)
SEEDS = (42, 100, 2024, 999, 1234)
BACKENDS = ("qnn", "classical")
DECODE_STRATEGIES = ("greedy", "sample")


def parse_options():
    parser = argparse.ArgumentParser(description="Audit and analyze the matched five-seed BatchNorm CVRP held-out evaluations.")
    parser.add_argument("--input", default="batchnorm_multiseed_cvrp_single_fixedtest_s1234")
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values)


def reported_cost(log_path):
    match = re.search(r"Average cost: ([0-9.eE+-]+)", log_path.read_text(errors="replace"))
    if not match:
        raise RuntimeError(f"missing average cost in {log_path}")
    return float(match.group(1))


def route_cost(instance, route):
    depot, locations, demands, capacity = instance
    if not route or route[-1] != 0:
        raise RuntimeError("route does not end at the depot")
    customers = [node for node in route if node != 0]
    n = len(locations)
    if sorted(customers) != list(range(1, n + 1)):
        raise RuntimeError("route does not visit every customer exactly once")
    load = 0
    previous = 0
    cost = 0.0
    for node in route:
        if node == 0:
            load = 0
            coordinate = depot
        else:
            load += demands[node - 1]
            if load > capacity + 1e-9:
                raise RuntimeError("route exceeds vehicle capacity")
            coordinate = locations[node - 1]
        previous_coordinate = depot if previous == 0 else locations[previous - 1]
        cost += math.dist(previous_coordinate, coordinate)
        previous = node
    return cost


def main():
    options = parse_options()
    repo = Path(__file__).resolve().parents[1]
    input_dir = repo / "results" / options.input
    manifest_path = input_dir / "evaluations.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    expected = {(seed, backend, size, strategy) for seed in SEEDS for backend in BACKENDS for size in SIZES for strategy in DECODE_STRATEGIES}
    actual = {(row["training_seed"], row["backend"], row["graph_size"], row["decode_strategy"]) for row in records}
    errors = []
    if any(row.get("decoding_seed") != 1234 for row in records):
        errors.append("one or more evaluations do not record decoding seed 1234")
    if len(records) != 80:
        errors.append(f"expected 80 manifest records, found {len(records)}")
    if actual != expected or len(actual) != len(records):
        errors.append(f"manifest membership mismatch: {len(actual)} unique expected {len(expected)}")

    datasets = {}
    means = {}
    total_results = 0
    for row in records:
        key = (row["training_seed"], row["backend"], row["graph_size"], row["decode_strategy"])
        artifact = Path(row["artifact"])
        log = Path(row["log"])
        checkpoint = Path(row["checkpoint"])
        dataset_path = Path(row["dataset"])
        for label, path in (("artifact", artifact), ("log", log), ("checkpoint", checkpoint), ("dataset", dataset_path)):
            if not path.is_file():
                errors.append(f"{key}: missing {label}: {path}")
        if any(not path.is_file() for path in (artifact, log, checkpoint, dataset_path)):
            continue
        expected_dataset = repo / f"data/vrp/vrp{row['graph_size']}_frontier_test1000_seed9876.pkl"
        if dataset_path.resolve() != expected_dataset.resolve():
            errors.append(f"{key}: wrong held-out dataset {dataset_path}")
        if dataset_path not in datasets:
            with dataset_path.open("rb") as source:
                datasets[dataset_path] = pickle.load(source)
        instances = datasets[dataset_path]
        with artifact.open("rb") as source:
            artifact_data = pickle.load(source)
        if not isinstance(artifact_data, tuple) or len(artifact_data) != 2:
            errors.append(f"{key}: malformed artifact container")
            continue
        results, parallelism = artifact_data
        if len(results) != 1000 or len(instances) != 1000:
            errors.append(f"{key}: expected 1000 results and instances, found {len(results)} and {len(instances)}")
            continue
        if not isinstance(parallelism, int) or parallelism < 1:
            errors.append(f"{key}: invalid parallelism {parallelism!r}")
        costs = []
        for index, (result, instance) in enumerate(zip(results, instances)):
            if not isinstance(result, tuple) or len(result) != 3:
                errors.append(f"{key}: malformed result {index}")
                break
            cost, route, duration = result
            cost = float(cost)
            if not math.isfinite(cost) or not math.isfinite(float(duration)):
                errors.append(f"{key}: non-finite result {index}")
                break
            try:
                recomputed = route_cost(instance, route)
            except RuntimeError as error:
                errors.append(f"{key}: infeasible result {index}: {error}")
                break
            if not math.isclose(cost, recomputed, rel_tol=2e-6, abs_tol=2e-5):
                errors.append(f"{key}: cost mismatch at result {index}: saved {cost}, recomputed {recomputed}")
                break
            costs.append(cost)
        if len(costs) != 1000:
            continue
        total_results += len(costs)
        artifact_mean = mean(costs)
        log_mean = reported_cost(log)
        if not math.isclose(artifact_mean, log_mean, rel_tol=1e-7, abs_tol=5e-6):
            errors.append(f"{key}: artifact mean {artifact_mean} differs from log {log_mean}")
        means[key] = artifact_mean

    paired_rows = []
    for size in SIZES:
        for strategy in DECODE_STRATEGIES:
            for seed in SEEDS:
                qnn = means.get((seed, "qnn", size, strategy))
                classical = means.get((seed, "classical", size, strategy))
                if qnn is None or classical is None:
                    continue
                paired_rows.append({
                    "n": size,
                    "decode_strategy": strategy,
                    "training_seed": seed,
                    "qnn_mean": qnn,
                    "classical_mean": classical,
                    "qnn_minus_classical": qnn - classical,
                    "qnn_minus_classical_pct": 100 * (qnn / classical - 1),
                })

    summary_rows = []
    for size in SIZES:
        for strategy in DECODE_STRATEGIES:
            relevant = [row for row in paired_rows if row["n"] == size and row["decode_strategy"] == strategy]
            differences = [row["qnn_minus_classical_pct"] for row in relevant]
            for backend in BACKENDS:
                values = [means[(seed, backend, size, strategy)] for seed in SEEDS if (seed, backend, size, strategy) in means]
                summary_rows.append({
                    "n": size,
                    "decode_strategy": strategy,
                    "backend": backend,
                    "seed_mean": mean(values),
                    "seed_sd": statistics.stdev(values),
                    "paired_pct_mean": mean(differences),
                    "paired_pct_sd": statistics.stdev(differences),
                    "paired_pct_min": min(differences),
                    "paired_pct_max": max(differences),
                })

    paired_path = input_dir / "paired-differences.tsv"
    with paired_path.open("w") as output:
        output.write("n\tdecode_strategy\ttraining_seed\tqnn_mean\tclassical_mean\tqnn_minus_classical\tqnn_minus_classical_pct\n")
        for row in paired_rows:
            output.write(f"{row['n']}\t{row['decode_strategy']}\t{row['training_seed']}\t{row['qnn_mean']:.9f}\t{row['classical_mean']:.9f}\t{row['qnn_minus_classical']:.9f}\t{row['qnn_minus_classical_pct']:.6f}\n")

    summary_path = input_dir / "five-seed-summary.tsv"
    with summary_path.open("w") as output:
        output.write("n\tdecode_strategy\tbackend\tseed_mean\tseed_sd\tpaired_pct_mean\tpaired_pct_sd\tpaired_pct_min\tpaired_pct_max\n")
        for row in summary_rows:
            output.write(f"{row['n']}\t{row['decode_strategy']}\t{row['backend']}\t{row['seed_mean']:.9f}\t{row['seed_sd']:.9f}\t{row['paired_pct_mean']:.6f}\t{row['paired_pct_sd']:.6f}\t{row['paired_pct_min']:.6f}\t{row['paired_pct_max']:.6f}\n")

    audit_lines = [
        "BatchNorm multiseed CVRP held-out evaluation audit",
        f"manifest records: {len(records)}",
        f"unique evaluation keys: {len(actual)}",
        f"artifacts checked: {len(means)}",
        f"instance results checked: {total_results}",
        "checks: complete five-seed/model/size/decode membership; artifact, log, checkpoint, and dataset presence; fixed seed-9876 dataset identity; 1000 results per artifact; finite costs and durations; exact customer coverage; capacity feasibility; recomputed Euclidean route cost; artifact/log mean agreement",
        f"result: {'PASS' if not errors else 'FAIL'}",
        "errors: none" if not errors else "errors:",
    ]
    if errors:
        audit_lines.extend(f"- {error}" for error in errors)
    audit_report = "\n".join(audit_lines) + "\n"
    (input_dir / "postcampaign-audit.txt").write_text(audit_report)
    print(audit_report)

    if not errors:
        print("Paired QNN-minus-classical differences (%):")
        for size in SIZES:
            for strategy in DECODE_STRATEGIES:
                relevant = [row for row in paired_rows if row["n"] == size and row["decode_strategy"] == strategy]
                differences = [row["qnn_minus_classical_pct"] for row in relevant]
                values = ", ".join(f"s{row['training_seed']}={row['qnn_minus_classical_pct']:+.4f}%" for row in relevant)
                print(f"n={size:3d} {strategy:6s}: mean={mean(differences):+.4f}%, SD={statistics.stdev(differences):.4f}%, range=[{min(differences):+.4f}%, {max(differences):+.4f}%]; {values}")
        print("\nFive-seed objective means ± SD:")
        for size in SIZES:
            for strategy in DECODE_STRATEGIES:
                fields = []
                for backend in BACKENDS:
                    values = [means[(seed, backend, size, strategy)] for seed in SEEDS]
                    fields.append(f"{backend}={mean(values):.6f} ± {statistics.stdev(values):.6f}")
                print(f"n={size:3d} {strategy:6s}: " + "; ".join(fields))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
