import numpy as np


class MDP:
    """
    Interface for more specific MDP learning algorithms such as
    (MC) Monte-Carlo Learning, (TD(0)) Temporal-Difference Learning and
    (TD(lambda))
    """

    def __init__(self, state_space_size, action_space_size, learning_rate, maturity_threshold, discount):
        """ Initialize action value function with zeros values. """

        self.iteration_count = 0
        self.maturity_threshold = maturity_threshold
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = learning_rate
        self.gamma = discount

        self.action_value_fn = np.zeros((state_space_size, action_space_size), np.float)

    def learn(self, history):
        """
        Update internal representation of action value function.

        :param history: A list of state, action, reward tuples.
        :return: void
        """

    def is_mature(self) -> bool:
        """
        Given the updates history which follows a given policy,
        is the current MDP learning mature?

        :return: True if we believe we know well enough the action
                 value function of the current MDP.
        """

        if self.iteration_count >= self.maturity_threshold:
            self.iteration_count = 0
            return True

        return False

    def get_action_value_function(self):
        return self.action_value_fn


class MonteCarlo(MDP):
    def __init__(self, state_space_size, action_space_size, learning_rate, maturity_threshold, discount):
        """Initialize the MDP."""
        super().__init__(state_space_size, action_space_size, learning_rate, maturity_threshold, discount)

    def learn(self, history):
        """
        Learning the action value function in Monte-Carlo follows this:

            Q(st, at) <-- Q(st, at) + alpha(Gt - Q(st, at))

        Change the value of a action value in the direction of the
        observed total reward for this state, action pair, following a
        given policy by a learning factor specified by the value of alpha.

        :param history: History is a list of tuples. The first element
                        in the tuple is the state, the second is the
                        action and the third one is the
            reward obtained from that state after one step. The order
            of elements in history list is such that the first element
            is the most recent event happened. I.e. The fist element
            in the list should be the before terminal state of an
            episode with the action applied and its obtained reward.
        """

        assert history is not None
        if not history:
            return

        total_rewards = np.zeros((self.state_space_size, self.action_space_size), np.float)
        cumulative_reward = 0

        """
        Backward computing of the total reward obtained from a state.
        
        Gt = Rt + gamma*Gt1
        """
        for (state, action, reward) in history:
            cumulative_reward += reward + self.gamma * cumulative_reward
            total_rewards[state][action] = cumulative_reward

        self.action_value_fn = self.action_value_fn + (self.alpha * (total_rewards - self.action_value_fn))

        self.iteration_count += 1


class TemporalDifference(MDP):
    """
    Implement Temporal Difference algorithm or TD(0)
    """

    def __init__(self, state_space_size, action_space_size, learning_rate, maturity_threshold, discount):
        """Initialize the MDP."""
        super().__init__(state_space_size, action_space_size, learning_rate, maturity_threshold, discount)

    def learn(self, history):
        """
        Learning the action value function in TD(0) follows this:

        Q(st, at) <-- Q(st, at) + alpha(Rt1 + gamma*Q(st1, at1) - Q(st, at))

        Change the value of a action value in the direction of the
        observed reward for this state, action pair, following a
        given policy, by a learning factor specified by alpha.

        :param history: History is a list of tuples. The first element
                        in the tuple is the state, the second is the
                        action and the third one is the
            reward obtained from that state after one step. The first
            element on the list is the most recent one.
        :return void
        """

        assert history is not None
        assert len(history) == 2

        # Remember the order of the list. The most recent event is first on the list.
        (state_t, action_t, reward_t1) = history[1]
        (state_t1, action_t1, reward_t2) = history[0]
        q = self.action_value_fn[state_t][action_t]
        q1 = self.action_value_fn[state_t1][action_t1]

        self.action_value_fn[state_t][action_t] = q + self.alpha * (reward_t1 + (self.gamma * q1) - q)

        self.iteration_count += 1


class TemporalDifferenceLambda(MDP):
    def __init__(self, state_space_size):
        """Initialize the state space."""
        super().__init__()

    def learn(self, history):
        """Do something"""

    def is_mature(self) -> bool:
        return False
