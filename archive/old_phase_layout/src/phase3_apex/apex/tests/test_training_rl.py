import unittest

import numpy as np

from apex.rl.bandit import LinUCBBandit, make_trace_context
from apex.rl.rewards import block_reward
from apex.training.data import DraftTrainingExample
from apex.training.losses import acceptance_utility_weights, flattening_penalty
from apex.traces import BlockTrace


class TrainingAndRLTest(unittest.TestCase):
    def test_acceptance_utility_weights_match_derivative(self):
        p = [0.8, 0.5, 0.25]
        weights = acceptance_utility_weights(p)
        self.assertAlmostEqual(weights[0], (0.8 + 0.8 * 0.5 + 0.8 * 0.5 * 0.25) / 0.8)
        self.assertAlmostEqual(weights[1], (0.8 * 0.5 + 0.8 * 0.5 * 0.25) / 0.5)
        self.assertAlmostEqual(weights[2], (0.8 * 0.5 * 0.25) / 0.25)

    def test_rejection_bonus_does_not_apply_to_full_acceptance(self):
        base = acceptance_utility_weights([0.8, 0.5], first_rejection=2, rejection_bonus=2.0)
        no_bonus = acceptance_utility_weights([0.8, 0.5], rejection_bonus=2.0)
        self.assertEqual(base, no_bonus)

    def test_flattening_penalty(self):
        self.assertAlmostEqual(flattening_penalty([0.8, 0.5, 0.25], [1.0, 2.0, 3.0]), 2.0 * 0.09 + 3.0 * 0.0625)

    def test_training_example_validation(self):
        example = DraftTrainingExample(
            input_ids=[1, 2],
            target_ids=[3, 4],
            regime=[1.0, 0.0],
            draft_len=2,
            first_rejection=1,
            accept_probs=[0.8, 0.5],
        )
        self.assertEqual(example.draft_len, 2)

    def test_reward_and_bandit_update(self):
        trace = BlockTrace(
            request_id="r",
            block_id=0,
            method="draft_sd",
            draft_len=4,
            first_rejection=3,
            latency_ms=10.0,
            entropy=0.5,
            repetition_density=0.4,
        )
        reward = block_reward(trace)
        context = make_trace_context(trace, max_depth=16)
        bandit = LinUCBBandit([1, 2, 4], context_dim=context.shape[0], alpha=0.1)
        before = bandit.choose(context)
        bandit.update(context, depth=4, reward=reward)
        restored = LinUCBBandit.from_state_dict(bandit.state_dict())
        after = bandit.choose(context)
        self.assertIsInstance(before.depth, int)
        self.assertEqual(after.depth, 4)
        self.assertEqual(restored.choose(context).depth, after.depth)
        self.assertTrue(np.isfinite(after.score))


if __name__ == "__main__":
    unittest.main()
