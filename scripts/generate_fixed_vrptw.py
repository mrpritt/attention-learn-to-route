#!/usr/bin/env python

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from problems.vrp.problem_vrp import VRPDataset
from utils.data_utils import save_dataset


CAPACITIES = {5: 10, 10: 20, 20: 30, 50: 40, 100: 50}
DISTRIBUTIONS = (
    'tw_r1', 'tw_r1_normal', 'tw_r1_solomon', 'tw_r1_uniform',
    'tw_r2', 'tw_r2_normal', 'tw_r2_solomon', 'tw_r2_uniform',
)


def serialize_instance(instance, capacity, distribution):
    return {
        'depot': instance['depot'].tolist(),
        'loc': instance['loc'].tolist(),
        'demand': (instance['demand'] * capacity).round().int().tolist(),
        'capacity': capacity,
        'ready': instance['ready'].tolist(),
        'due': instance['due'].tolist(),
        'service_time': instance['service_time'].tolist(),
        'horizon': instance['horizon'].item(),
        'distribution': distribution,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate a fixed dataset from the online VRPTW distribution')
    parser.add_argument('--graph_size', type=int, required=True, choices=CAPACITIES)
    parser.add_argument('--dataset_size', type=int, default=1000)
    parser.add_argument('--distribution', choices=DISTRIBUTIONS, default='tw_r1')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = VRPDataset(
        size=args.graph_size,
        num_samples=args.dataset_size,
        distribution=args.distribution,
    )
    capacity = CAPACITIES[args.graph_size]
    save_dataset(
        [serialize_instance(instance, capacity, args.distribution) for instance in dataset],
        args.output,
    )
    print(f'Saved {len(dataset)} {args.distribution} instances to {args.output}')


if __name__ == '__main__':
    main()
