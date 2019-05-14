import pypownet.environment
import pypownet.agent as agent
import agents.model as model
import agents.policy as policy

import numpy as np


class CustomAgent(agent.Agent):
    """
    The template to be used to create an agent.
    This class overloads the feed_reward function of its mother class.
    """

    def __init__(self, environment):
        """ Initialize a new agent. """

        super().__init__(environment)

    def act(self, observation):
        """Produces an action given an observation of the environment.

        Takes as argument an observation of the current state, and returns the chosen action of class Action or np
        array."""
        pass

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

    Attributes
    ----------
    mdp:
    policy:
    history:

    Methods
    -------
    act(observation)
    feed_reward()

    learn()
    """

    def __init__(self, environment):
        assert isinstance(environment, pypownet.environment.RunEnv)
        super().__init__(environment)

        self.mdp = model.MonteCarlo(environment.observation_space.number_power_lines)
        self.policy = policy.Policy()
        self.history = list()

    def act(self, observation):
        observation = self.environment.observation_space.array_to_observation(observation)
        assert isinstance(observation, pypownet.environment.Observation)

        # Implement your policy here.
        action = self.environment.action_space.get_do_nothing_action()

        # Use the policy instance to get the best action for this state
        # action = policy.get_best_action(observation)

        return action

    def learn(self):
        """
        Learn from the observed interaction with the environment.

        :return: void
        """

        self.mdp.learn(self.history)

        if self.mdp.is_mature():
            self.policy.improve()

    def log_history(self, reward):
        """
        Create a history of visited states and the obtained reward.
        Once the episode terminated, this history serves to build
        the total reward for every of visited states.

        :return: void
        """

    def feed_return(self, action, consequent_observation, rewards_list, done):
        """
        Process the obtained reward for the last applied action.

        :param action:
        :param consequent_observation:
        :param rewards_list:
        :param done:
        :return:
        """

        self.log_history(sum(rewards_list))

        if done:
            self.learn()
