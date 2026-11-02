#!/usr/bin/env python3
"""Integrity tests for frozen labels, model matrices and subject-level folds."""

from __future__ import annotations

import unittest

import numpy as np

from training.dataset import fold_indices, load_dataset, verify_assets


class TrainingAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checks = verify_assets()
        cls.x, cls.y, cls.ahi, cls.metadata = load_dataset(verify=False)

    def test_checksums(self):
        self.assertTrue(all(self.checks.values()))

    def test_shape_and_identity(self):
        self.assertEqual(self.x.shape, (5785, 479))
        self.assertEqual(self.metadata["subject_id"].nunique(), 5785)
        self.assertFalse(self.metadata["subject_id"].duplicated().any())

    def test_target_thresholds(self):
        expected = np.digitize(self.ahi.to_numpy(), [5.0, 15.0, 30.0])
        np.testing.assert_array_equal(expected, self.y.to_numpy())

    def test_folds_are_disjoint_and_complete(self):
        seen = np.zeros(len(self.metadata), dtype=np.int8)
        for fold in range(5):
            train, test = fold_indices(self.metadata, fold)
            self.assertEqual(np.intersect1d(train, test).size, 0)
            seen[test] += 1
        np.testing.assert_array_equal(seen, np.ones_like(seen))

    def test_no_forbidden_columns(self):
        forbidden = ("target", "label", "ahi", "event", "arousal", "sleep_stage")
        offenders = [column for column in self.x if any(token in column.lower() for token in forbidden)]
        self.assertEqual(offenders, [])

    def test_allnight_is_finite(self):
        self.assertTrue(np.isfinite(self.x.to_numpy()).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
