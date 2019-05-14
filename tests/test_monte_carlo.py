import unittest
import numpy as np
import agents.model as model


class TestMonteCarlo(unittest.TestCase):
    """
    Test the Monte-Carlo learning algorithm.
    """

    def test_init_state_space_size_initialization(self):
        mdp = model.MonteCarlo(10, 0.0, 1)
        self.assertEqual(mdp.value_fn.size, 10)

    def test_init_value_fn_initialization(self):
        mdp = model.MonteCarlo(10, 0.0, 0)
        comparing_array = np.zeros(10, float)
        self.assertTrue(np.array_equal(comparing_array, mdp.value_fn))

    def test_learn(self):
        mdp = model.MonteCarlo(5, 0.1, 5)
        history = ((1, 1.0), (4, 1.5), (0, 0.9))

        mdp.learn(history)

        self.assertEqual(mdp.value_fn[0], 0.34)
        self.assertEqual(mdp.value_fn[1], 0.1)
        self.assertEqual(mdp.value_fn[2], 0.0)
        self.assertEqual(mdp.value_fn[3], 0.0)
        self.assertEqual(mdp.value_fn[4], 0.25)

    def test_is_mature_returns_false_if_untrained(self):
        mdp = model.MonteCarlo(2, 0.1, 1)
        self.assertFalse(mdp.is_mature())

    def test_is_mature_returns_true_if_trained(self):
        mdp = model.MonteCarlo(5, 0.1, 1)
        history = ((1, 1.0), (4, 1.5), (0, 0.9))

        mdp.learn(history)

        self.assertTrue(mdp.is_mature())

    # TODO: terminate testing extreme cases for MC algorithm
