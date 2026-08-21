import unittest

from apex.survival import (
    accepted_length_pmf,
    expected_accepted_length,
    expected_accepted_length_exact,
    marginal_depth_value,
    survival_curve,
)


class SurvivalMathTest(unittest.TestCase):
    def test_pmf_matches_corrected_formula(self):
        p = [0.8, 0.5, 0.25]
        pmf = accepted_length_pmf(p)
        self.assertAlmostEqual(pmf[0], 0.2)
        self.assertAlmostEqual(pmf[1], 0.8 * 0.5)
        self.assertAlmostEqual(pmf[2], 0.8 * 0.5 * 0.75)
        self.assertAlmostEqual(pmf[3], 0.8 * 0.5 * 0.25)
        self.assertAlmostEqual(sum(pmf), 1.0)

    def test_exact_expectation_equals_survival_sum(self):
        p = [0.9, 0.8, 0.7, 0.6]
        self.assertAlmostEqual(expected_accepted_length_exact(p), expected_accepted_length(p))
        self.assertAlmostEqual(expected_accepted_length(p), 0.9 + 0.9 * 0.8 + 0.9 * 0.8 * 0.7 + 0.9 * 0.8 * 0.7 * 0.6)

    def test_survival_curve_uses_first_token_probability_at_zero(self):
        self.assertEqual(survival_curve([0.4, 0.5]), [0.4, 0.2])

    def test_marginal_depth_value(self):
        self.assertAlmostEqual(marginal_depth_value([0.8, 0.5], 0.25), 0.1)


if __name__ == "__main__":
    unittest.main()
