import numpy as np


class MDP:
    """
    Interface for more specific MDP learning algorithms such as
    (MC) Monte-Carlo Learning, (TD(0)) Temporal-Difference Learning and
    (TD(lambda))
    """

    def __init__(self):
        """ Initialize value function with random values """

    def learn(self, history):
        """
        Update internal representation of value function.

        :param history: A list of state reward tuples
        :return: void
        """

    def is_mature(self) -> bool:
        """
        Given the updates history which follows a given policy,
        is the current MDP learning mature?
        :return: True if we believe we know well enough the value
                 function of the current MDP
        """
        return False


class MonteCarlo(MDP):
    def __init__(self, state_space_size, alpha, maturity_threshold):
        """Initialize the state space."""
        super().__init__()

        self.value_fn = np.zeros(state_space_size, float)
        self.alpha = alpha
        self.state_space_size = state_space_size
        self.iteration_count = 0
        self.maturity_threshold = maturity_threshold


    def learn(self, history):
        """
        Learning the state value function in Monte-Carlo follows this:

            V(st) <-- V(st) + alpha(Gt - V(st))

        Change the value of a state in the direction of the observed
        total reward for this state following a given policy by a
        factor specified by the value of alpha.

        :param history: History is a list of tuples. The first element
                        in the tuple is the state and the second is the
            reward obtained from that state after one step. The order
            of elements in history list is such that the first element
            is the most recent event happened. I.e. The fist element
            in the list should be the terminal state of en episode with
            its obtained reward.
        """

        total_rewards = np.zeros(self.state_space_size)
        state_total_reward = 0

        # Backward computing of the total reward obtained from a state
        for (state, reward) in history:
            state_total_reward += reward
            total_rewards[state] = state_total_reward

        self.value_fn = self.value_fn + (self.alpha * (total_rewards - self.value_fn))

        self.iteration_count += 1

    def is_mature(self) -> bool:
        """
        Is the value function learned enough for the current policy.
        In the current implementation our metric is simply based on a
        counter. If we run enough iterations than, the value function
        should be mature enough.

        :return: True if the number of learning iterations is bigger
                 than the specified threshold.
        """
        return self.iteration_count > self.maturity_threshold


class TemporalDifference(MDP):
    def __init__(self, state_space_size):
        """Initialize the state space."""
        super().__init__()

    def learn(self, history):
        """Do something"""

    def is_mature(self) -> bool:
        return False


class TemporalDifferenceLambda(MDP):
    def __init__(self, state_space_size):
        """Initialize the state space."""
        super().__init__()

    def learn(self, history):
        """Do something"""

    def is_mature(self) -> bool:
        return False
