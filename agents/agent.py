import pypownet.environment
import pypownet.agent as agent
import agents.model as model
import agents.policy as policy
import agents.wrapper as wrapper

import numpy as np


class CustomAgent(agent.Agent):
    """
    The template to be used to create an agent.
    This class overloads the feed_reward function of its mother class.
    """

    def __init__(self, environment):
        """ Initialize a new agent. """

        super().__init__(environment)

        """List of (state, action, reward) tuples for all states visited during an episode."""
        self.history = list()
        self.last_state = -1
        self.last_action = -1

        """Learning rate. Used in Monte-Carlo action value learning."""
        self.alpha = 0.1
        """How many iteration to learn the action-value before policy improvement."""
        self.mdp_iteration = 10
        """Discount factor for total return computing."""
        self.gamma = 0.8
        """The probability to explore a new action instead of exploiting what we now."""
        self.epsilon = 0.1

        self.state_space_size = np.power(2, environment.observation_space.number_power_lines)
        self.action_space_size = environment.action_space.prods_switches_subaction_length + \
                                 environment.action_space.loads_switches_subaction_length + \
                                 environment.action_space.lines_or_switches_subaction_length + \
                                 environment.action_space.lines_ex_switches_subaction_length

        self.mdp = None
        self.policy = None

    def act(self, observation):
        """
        Given the observation return an action to apply.

        :param observation: The environment observations.
        :return: Action to apply to the environment.
        """

        assert self.mdp is not None
        assert self.policy is not None

        observation = self.environment.observation_space.array_to_observation(observation)

        self.last_state = wrapper.observation_to_state(observation)
        self.last_action = self.policy.get_action(self.last_state)

        do_nothing_action_array = self.environment.action_space.get_do_nothing_action()
        do_nothing_action = self.environment.action_space.array_to_action(do_nothing_action_array)

        split_substation_action = wrapper.agents_action_to_envs_action(do_nothing_action, self.last_action)

        return split_substation_action

    def learn(self):
        """
        Learn from the observed interaction with the environment.

        :return: void
        """

        self.mdp.learn(self.history)
        self.history.clear()

        if self.mdp.is_mature():
            self.policy.improve(self.mdp.get_action_value_function())

    def feed_return(self, action, consequent_observation, rewards_as_list, done):
        """
        This function has the same purpose as the feed_reward from
        super class. The reason of this adding is to inform the agent
        about the end of an episode.
        Override this method in order to learn from the reward obtained
        from the environment after the application of the last action.

        :param action: The applied action.
        :param consequent_observation: Observation after the
                 application of the action.
        :param rewards_as_list: A list of rewards for the last step.
        :param done: True if is the end of en episode. False otherwise.
        :return: void
        """

        pass


class PolicyIteration(CustomAgent):
    """
    An implementation of Policy Iteration algorithm for Reinforcement
    Learning

    A policy iteration algorithm implies two steps:
          1. Policy Evaluation - finds out the value function of an MDP
             given the MDP and a policy.
          2. Policy Improvement - finds a new policy at least equal or
             better than the old one.
    """

    def __init__(self, environment):
        assert isinstance(environment, pypownet.environment.RunEnv)
        super().__init__(environment)

        """For this test use MonteCarlo to learn the action-value function."""
        self.mdp = model.MonteCarlo(self.state_space_size, self.action_space_size, self.alpha, self.mdp_iteration,
                                    self.gamma)
        """For this test use EpsilonGreedy for policy improvement."""
        self.policy = policy.EpsilonGreedy(self.state_space_size, self.action_space_size, self.epsilon)

    def log_history(self, state, action, reward):
        """
        Create a history of visited states and the obtained reward.
        Once the episode terminated, this history serves to build
        the total reward for every of visited states.

        :param state: the state for which we know the immediate reward.
        :param action: the action taken in last state.
        :param reward: the reward obtained for the last step.
        :return: void
        """

        self.history.insert(0, (state, action, reward))

    def feed_return(self, action, consequent_observation, rewards_list, done):
        """
        Process the obtained reward for the last applied action.

        :param action:
        :param consequent_observation:
        :param rewards_list:
        :param done:
        :return:
        """

        self.log_history(self.last_state, self.last_action, sum(rewards_list) + 5)

        if done:
            self.learn()


class Sarsa(CustomAgent):
    """
    Implement an agent using SARSA algorithm.
    """

    def __init__(self, environment):
        assert isinstance(environment, pypownet.environment.RunEnv)
        super().__init__(environment)

        """For this test use TD(0) to learn the action-value function."""
        self.mdp = model.TemporalDifference(self.state_space_size, self.action_space_size, self.alpha,
                                            self.mdp_iteration, self.gamma)
        """For this test use EpsilonGreedy for policy improvement."""
        self.policy = policy.EpsilonGreedy(self.state_space_size, self.action_space_size, self.epsilon)

    def feed_return(self, action, consequent_observation, rewards_as_list, done):
        """
        Process the obtained reward for the last applied action.

        :param action:
        :param consequent_observation:
        :param rewards_as_list:
        :param done:
        :return:
        """

        consequent_observation = self.environment.observation_space.array_to_observation(consequent_observation)

        # The history follows the format (St, At, Rt1, St1, At1, Rt2, ...)
        state_t1 = wrapper.observation_to_state(consequent_observation)
        action_t1 = self.policy.get_action(state_t1)
        reward_t2 = 0

        self.history.append((state_t1, action_t1, reward_t2))

        state_t = self.last_state
        action_t = self.last_action
        reward_t1 = sum(rewards_as_list) + 5

        self.history.append((state_t, action_t, reward_t1))

        self.learn()


class QLearning(CustomAgent):
    """
    Implement an agent using Q-learning algorithm which is an
    off-policy TD control algorithm. Q-learning estimates a
    state-action value function for a target policy that
    deterministically selects the action of highest value.
    """

    def __init__(self, environment):
        assert isinstance(environment, pypownet.environment.RunEnv)
        super().__init__(environment)

        """For this test use TD(0) to learn the action-value function."""
        self.mdp = model.TemporalDifference(self.state_space_size, self.action_space_size, self.alpha,
                                            self.mdp_iteration, self.gamma)
        """For this test use EpsilonGreedy for policy improvement."""
        self.policy = policy.EpsilonGreedy(self.state_space_size, self.action_space_size, self.epsilon)

    def feed_return(self, action, consequent_observation, rewards_as_list, done):
        """
        Process the obtained reward for the last applied action. For
        the implementation of QLearning we use TD(0) learning algo.
        QLearning follows this expression:

        Q(S, A) <- Q(S, A) + alpha(R + gamma*max Q(S', A') - Q(S, A))

        If if the next state S' is a terminal state then Q(S',: ) = 0

        :param action:
        :param consequent_observation:
        :param rewards_as_list:
        :param done:
        :return: void
        """

        consequent_observation = self.environment.observation_space.array_to_observation(consequent_observation)

        # The history follows the format (St, At, Rt1, St1, At1, Rt2, ...)
        # Find out max Q(St1, At1)
        state_t1 = wrapper.observation_to_state(consequent_observation)
        action_t1 = np.argmax(self.mdp.get_action_value_function()[state_t1])
        reward_t2 = 0

        self.history.append((state_t1, action_t1, reward_t2))

        state_t = self.last_state
        action_t = self.last_action
        reward_t1 = sum(rewards_as_list) + 5

        self.history.append((state_t, action_t, reward_t1))

        # If we are in terminal state set action-state values to zero
        if done:
            (self.mdp.get_action_value_function()[state_t1]).fill(0)

        self.learn()
