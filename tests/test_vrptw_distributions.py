import unittest

import torch

from problems.vrp.problem_vrp import VRPDataset, make_instance
from scripts.generate_fixed_vrptw import serialize_instance


class VRPTWDistributionTest(unittest.TestCase):
    @staticmethod
    def generate(distribution, seed=4321, samples=8, size=20):
        torch.manual_seed(seed)
        return VRPDataset(size=size, num_samples=samples, distribution=distribution)

    def assert_same_dataset(self, left, right):
        self.assertEqual(len(left), len(right))
        for left_instance, right_instance in zip(left, right):
            self.assertEqual(left_instance.keys(), right_instance.keys())
            for key in left_instance:
                torch.testing.assert_close(left_instance[key], right_instance[key])

    def test_plain_and_solomon_names_alias_normal_distribution(self):
        for family in ('r1', 'r2'):
            normal = self.generate(f'tw_{family}_normal')
            self.assert_same_dataset(normal, self.generate(f'tw_{family}'))
            self.assert_same_dataset(normal, self.generate(f'tw_{family}_solomon'))

    def test_uniform_distribution_remains_explicit_and_distinct(self):
        normal = self.generate('tw_r1_normal')
        uniform = self.generate('tw_r1_uniform')
        self.assertFalse(torch.equal(normal[0]['ready'], uniform[0]['ready']))
        self.assertFalse(torch.equal(normal[0]['due'], uniform[0]['due']))

    def test_generated_windows_are_individually_feasible(self):
        for distribution in ('tw_r1_normal', 'tw_r1_uniform', 'tw_r2_normal', 'tw_r2_uniform'):
            for instance in self.generate(distribution):
                depot_distance = (instance['loc'] - instance['depot']).norm(p=2, dim=-1)
                self.assertTrue(torch.all(instance['ready'] >= 0))
                self.assertTrue(torch.all(instance['ready'] <= instance['due']))
                self.assertTrue(torch.all(instance['due'] + instance['service_time'] + depot_distance <= instance['horizon'] + 1e-6))

    def test_fixed_serialization_round_trips(self):
        original = self.generate('tw_r1_normal', samples=1)[0]
        restored = make_instance(serialize_instance(original, capacity=30, distribution='tw_r1_normal'))
        for key in ('depot', 'loc', 'demand', 'ready', 'due', 'service_time', 'horizon'):
            torch.testing.assert_close(original[key], restored[key])

    def test_unknown_tw_distribution_is_rejected(self):
        with self.assertRaises(ValueError):
            self.generate('tw_r1_typo')


if __name__ == '__main__':
    unittest.main()
