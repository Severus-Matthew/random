import unittest

from apex.controllers import CostModel, SurvivalDepthController, ThresholdDepthController, choose_regime
from apex.hazard import BetaHazardEstimator
from apex.metrics import summarize_traces
from apex.replay import replay_depth_controller
from apex.traces import BlockTrace


class HazardAndControllerTest(unittest.TestCase):
    def test_hazard_estimator_respects_censoring(self):
        traces = [
            BlockTrace(request_id="a", block_id=0, method="ngram_sd", draft_len=4, first_rejection=2),
            BlockTrace(request_id="b", block_id=0, method="ngram_sd", draft_len=4, first_rejection=4),
            BlockTrace(request_id="c", block_id=0, method="ngram_sd", draft_len=4, first_rejection=0),
        ]
        estimator = BetaHazardEstimator(max_depth=4, alpha=1.0, beta=1.0)
        estimator.update_many(traces)
        estimate = estimator.estimate()

        self.assertEqual(estimate.exposures, (3, 2, 2, 1))
        self.assertEqual(estimate.accepts, (2.0, 2.0, 1.0, 1.0))
        self.assertAlmostEqual(estimate.accept_probs[0], 3.0 / 5.0)
        self.assertAlmostEqual(estimate.accept_probs[3], 2.0 / 3.0)

    def test_survival_depth_controller_is_cost_aware(self):
        controller = SurvivalDepthController(
            [1, 2, 4],
            cost_model=CostModel(verifier_ms=1.0, draft_ms_per_token=0.5),
        )
        decision = controller.choose([0.9, 0.8, 0.1, 0.1])
        self.assertIn(decision.depth, {1, 2})
        self.assertGreater(decision.expected_len, 0.0)

    def test_threshold_controller_stops_on_low_survival(self):
        controller = ThresholdDepthController(min_depth=1, max_depth=4, min_marginal_value=0.2)
        decision = controller.choose([0.9, 0.8, 0.1, 0.1])
        self.assertEqual(decision.depth, 2)

    def test_regime_rule(self):
        self.assertEqual(choose_regime({"entropy": 0.2, "repetition_density": 0.8}), "copy_like")
        self.assertEqual(choose_regime({"entropy": 4.0, "repetition_density": 0.0}), "conservative")
        self.assertEqual(choose_regime({"entropy": 2.0, "repetition_density": 0.1}), "neural")

    def test_summary_and_replay(self):
        traces = [
            BlockTrace(request_id="a", block_id=0, method="ngram_sd", draft_len=2, first_rejection=2, latency_ms=10.0),
            BlockTrace(request_id="a", block_id=1, method="ngram_sd", draft_len=2, first_rejection=1, latency_ms=20.0),
        ]
        summary = summarize_traces(traces)
        self.assertEqual(summary.total_drafted_tokens, 4)
        self.assertEqual(summary.total_accepted_tokens, 3)
        self.assertAlmostEqual(summary.acceptance_rate, 0.75)
        self.assertAlmostEqual(summary.mean_latency_ms, 15.0)
        self.assertAlmostEqual(summary.p95_latency_ms, 20.0)

        decisions = replay_depth_controller(traces, max_depth=2, candidate_depths=(1, 2))
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0].chosen_depth_before_update, 2)


if __name__ == "__main__":
    unittest.main()
