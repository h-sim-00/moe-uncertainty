"""CPU-only unit tests for the paired OoD comparison statistics."""

import unittest

import numpy as np

from evaluate_ilv_ood_arms import (
    delta_verdict,
    domain_sampling_seed,
    paired_bootstrap_delta,
)
from uq_stats import auroc


class PairedDeltaTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0] * 50 + [1] * 50)
        self.chance = np.full(100, 0.5)
        self.perfect = np.array([0.1] * 50 + [0.9] * 50)
        self.inverted = np.array([0.9] * 50 + [0.1] * 50)

    def test_detects_better_arm(self):
        d = paired_bootstrap_delta(
            self.labels, self.chance, self.perfect, auroc, n_boot=200, seed=7
        )
        self.assertAlmostEqual(d["point"], 0.5)
        self.assertGreater(d["lo"], 0.0)
        self.assertEqual(delta_verdict(d), "BETTER with explanations")

    def test_detects_worse_arm(self):
        d = paired_bootstrap_delta(
            self.labels, self.chance, self.inverted, auroc, n_boot=200, seed=7
        )
        self.assertAlmostEqual(d["point"], -0.5)
        self.assertLess(d["hi"], 0.0)
        self.assertEqual(delta_verdict(d), "WORSE with explanations")

    def test_identical_scores_are_inconclusive(self):
        d = paired_bootstrap_delta(
            self.labels, self.perfect, self.perfect, auroc, n_boot=100, seed=7
        )
        self.assertEqual(d["point"], 0.0)
        self.assertEqual(d["lo"], 0.0)
        self.assertEqual(d["hi"], 0.0)
        self.assertEqual(delta_verdict(d), "NO CLEAR DIFFERENCE")

    def test_rejects_misaligned_shapes(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_delta(
                self.labels, self.chance[:-1], self.perfect, auroc, n_boot=10, seed=1
            )

    def test_domain_sampling_seeds_are_stable_and_distinct(self):
        self.assertEqual(domain_sampling_seed(42, 0), 42)
        self.assertEqual(domain_sampling_seed(42, 1), 1_000_045)
        self.assertEqual(domain_sampling_seed(42, 3), domain_sampling_seed(42, 3))


if __name__ == "__main__":
    unittest.main()
