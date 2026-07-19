import unittest
import warnings

import numpy as np

from amplify_bbopt_ext.initial_sampling import (
    MarginalCoverageWarning,
    SobolBalanceWarning,
    lhs_initial_samples,
    lhs_unit_samples,
    map_unit_samples_to_discrete,
    sobol_initial_samples,
    sobol_unit_samples,
    warn_if_incomplete_marginal_coverage,
)


class ContinuousInitialSamplingTest(unittest.TestCase):
    def test_lhs_is_reproducible_and_scaled(self) -> None:
        lower = np.array([-2.0, 10.0])
        upper = np.array([2.0, 20.0])
        first = lhs_initial_samples(lower, upper, 8, seed=42)
        second = lhs_initial_samples(lower, upper, 8, seed=42)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (8, 2))
        self.assertTrue(np.all(first >= lower))
        self.assertTrue(np.all(first <= upper))

    def test_sobol_power_of_two_is_reproducible_without_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = sobol_initial_samples([0.0, -1.0], [1.0, 1.0], 8, seed=7)
            second = sobol_initial_samples([0.0, -1.0], [1.0, 1.0], 8, seed=7)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(any(issubclass(item.category, SobolBalanceWarning) for item in caught))

    def test_sobol_arbitrary_count_warns_but_returns_samples(self) -> None:
        with self.assertWarns(SobolBalanceWarning):
            samples = sobol_unit_samples(3, 7, seed=1, scramble=True)
        self.assertEqual(samples.shape, (7, 3))

    def test_invalid_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            lhs_initial_samples([0.0], [0.0], 4, seed=1)
        with self.assertRaises(ValueError):
            lhs_unit_samples(0, 4, seed=1)


class DiscreteMappingTest(unittest.TestCase):
    def test_numeric_candidates_are_mapped_without_one_hot_encoding(self) -> None:
        unit = np.array([[0.0, 0.20], [0.49, 0.51], [0.99, 0.99]])
        values = [[10, 20], [1, 2, 3]]
        mapped = map_unit_samples_to_discrete(unit, values)

        np.testing.assert_array_equal(mapped, np.array([[10, 1], [10, 2], [20, 3]]))

    def test_mixed_candidates_preserve_python_values(self) -> None:
        unit = np.array([[0.1, 0.1], [0.9, 0.9]])
        mapped = map_unit_samples_to_discrete(unit, [[1, 2], ["red", "blue"]])

        self.assertEqual(mapped.dtype, np.dtype(object))
        self.assertEqual(mapped.tolist(), [[1, "red"], [2, "blue"]])

    def test_incomplete_coverage_warns_instead_of_raising(self) -> None:
        samples = np.array([[0, "red"], [0, "blue"]], dtype=object)
        values = [[0, 1], ["red", "blue"]]

        with self.assertWarns(MarginalCoverageWarning):
            report = warn_if_incomplete_marginal_coverage(samples, values)

        self.assertFalse(report.is_complete)
        self.assertEqual(report.missing[0].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
