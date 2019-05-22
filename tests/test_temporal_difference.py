import unittest
import numpy as np
import agents.model as model


class TestTemporalDifference(unittest.TestCase):
    """
    Test the TD(0) learning algorithm.
    """

    def test_init_state_space_size_initialization(self):
        mdp = model.TemporalDifference(10, 1, 0.0, 1, 0.5)
        self.assertEqual(mdp.action_value_fn.size, 10)

    def test_init_value_fn_initialization(self):
        mdp = model.TemporalDifference(10, 2, 0.0, 0, 0.5)
        comparing_array = np.zeros((10, 2), float)
        self.assertTrue(np.array_equal(comparing_array, mdp.action_value_fn))

    def test_learn(self):
        mdp = model.TemporalDifference(2, 2, 1.0, 5, 0.5)
        history = ((1, 0, 0.0), (0, 1, 1.5))

        mdp.learn(history)

        self.assertEqual(mdp.action_value_fn[0][0], 0.0)
        self.assertEqual(mdp.action_value_fn[1][0], 0.0)
        self.assertEqual(mdp.action_value_fn[0][1], 1.5)
        self.assertEqual(mdp.action_value_fn[1][1], 0.0)

        history = ((0, 1, 0.0), (1, 0, 1.5))

        mdp.learn(history)

        self.assertEqual(mdp.action_value_fn[0][0], 0.0)
        self.assertEqual(mdp.action_value_fn[1][0], 2.25)
        self.assertEqual(mdp.action_value_fn[0][1], 1.5)
        self.assertEqual(mdp.action_value_fn[1][1], 0.0)

    def test_learn_none_history(self):
        mdp = model.TemporalDifference(1, 2, 0.4, 3, 0.6)

        try:
            mdp.learn(None)
            self.fail("The input is not check for none.")
        except AssertionError:
            pass

    def test_learn_empty_history_does_nothing(self):
        mdp = model.TemporalDifference(1, 2, 0.4, 3, 0.6)

        try:
            mdp.learn([])
            self.fail()
        except AssertionError:
            self.assertEqual(mdp.action_value_fn[0][0], 0.0)
            self.assertEqual(mdp.action_value_fn[0][1], 0.0)

    def test_is_mature_returns_false_if_untrained(self):
        mdp = model.TemporalDifference(2, 2, 0.1, 1, 0.5)
        self.assertFalse(mdp.is_mature())

    def test_is_mature_returns_true_if_trained(self):
        mdp = model.TemporalDifference(5, 2, 0.1, 1, 0.5)
        history = ((1, 0, 1.0), (4, 0, 1.5))

        mdp.learn(history)

        self.assertTrue(mdp.is_mature())
