import unittest
import numpy as np
import agents.model as model


class TestMonteCarlo(unittest.TestCase):
    """
    Test the Monte-Carlo learning algorithm.
    """

    def test_init_state_space_size_initialization(self):
        mdp = model.MonteCarlo(10, 1, 0.0, 1, 0.5)
        self.assertEqual(mdp.action_value_fn.size, 10)

    def test_init_value_fn_initialization(self):
        mdp = model.MonteCarlo(10, 2, 0.0, 0, 0.5)
        comparing_array = np.zeros((10, 2), float)
        self.assertTrue(np.array_equal(comparing_array, mdp.action_value_fn))

    def test_learn(self):
        mdp = model.MonteCarlo(5, 2, 1.0, 5, 0.5)
        history = ((1, 0, 1.0), (4, 1, 1.5), (0, 0, 0.9))

        mdp.learn(history)

        self.assertEqual(mdp.action_value_fn[0][0], 5.4)
        self.assertEqual(mdp.action_value_fn[1][0], 1.0)
        self.assertEqual(mdp.action_value_fn[2][0], 0.0)
        self.assertEqual(mdp.action_value_fn[3][0], 0.0)
        self.assertEqual(mdp.action_value_fn[4][0], 0.0)
        self.assertEqual(mdp.action_value_fn[0][1], 0.0)
        self.assertEqual(mdp.action_value_fn[1][1], 0.0)
        self.assertEqual(mdp.action_value_fn[2][1], 0.0)
        self.assertEqual(mdp.action_value_fn[3][1], 0.0)
        self.assertEqual(mdp.action_value_fn[4][1], 3.0)

    def test_learn_none_history(self):
        mdp = model.MonteCarlo(1, 2, 0.4, 3, 0.6)

        try:
            mdp.learn(None)
            self.fail("The input is not check for none.")
        except AssertionError:
            pass

    def test_learn_empty_history_does_nothing(self):
        mdp = model.MonteCarlo(1, 2, 0.4, 3, 0.6)

        mdp.learn([])

        self.assertEqual(mdp.action_value_fn[0][0], 0.0)
        self.assertEqual(mdp.action_value_fn[0][1], 0.0)

    def test_is_mature_returns_false_if_untrained(self):
        mdp = model.MonteCarlo(2, 2, 0.1, 1, 0.5)
        self.assertFalse(mdp.is_mature())

    def test_is_mature_returns_true_if_trained(self):
        mdp = model.MonteCarlo(5, 2, 0.1, 1, 0.5)
        history = ((1, 0, 1.0), (4, 0, 1.5), (0, 1, 0.9))

        mdp.learn(history)

        self.assertTrue(mdp.is_mature())
