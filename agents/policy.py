import numpy as np


class Policy:
    def __init__(self):
        """ Initialize with random values """

    def get_action(self, state: int) -> int:
        """
        Get the best action for this state

        :param state: Each bit in state variable represents the state
                      of a line.
        :return: Each bit in returned value is the state of a
                 substation.
        """

    def improve(self, action_value_fn):
        """
        Returns another instance of Policy class which is at least
        as good as this one.

        :param action_value_fn: The action value array.
        :return: void
        """

    def is_mature(self) -> bool:
        """
        Given the improvements history, is this policy mature?
        I.e. is this policy the optimal policy for this MDP?

        :return: True if we believe the policy converged to the optimal
                 policy for the MDP on which it was trained.
        """
        return False


class EpsilonGreedy(Policy):
    """
    An implementation of epsilon greedy policy.
    """

    def __init__(self, state_space_size, action_space_size, epsilon):
        """
        Initialize the policy randomly. As a MDP has at least one
        deterministic optimal policy we map each state to only one
        action. Eventually, this action will be changed during the
        policy improvement step.

        :param state_space_size: The size of the state space.
        :param action_space_size:  The size of the action space.
        :param epsilon: The exploration probability.
        """

        super().__init__()

        self.policy = np.random.randint(0, action_space_size, state_space_size)
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.epsilon = epsilon

    def get_action(self, state: int) -> int:
        """
        Randomly choose to explore or exploit. If exploration is chosen
        then choose randomly uniformly an action from the action space.

        :param state: The state in which the environment is know.
        :return: An action conform to the policy and the exploration
                 factor.
        """
        if np.random.random_sample() > self.epsilon:
            return self.policy[state]

        return np.random.randint(self.action_space_size)

    def improve(self, action_value_fn):
        """
        Given the action value array choose the action with the maximum
        value for each state.

        :param action_value_fn: An array of size state_space_size by
                                action_space_size containing the value
            of each action state pair. Normally this array comes from
            the model which can evaluate it using Monte Carlo or other
            learning algorithm.
        :return: void
        """

        self.policy = np.argmax(action_value_fn, axis=1)
