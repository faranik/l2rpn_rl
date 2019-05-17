from pypownet.environment import RunEnv
from pypownet.runner import Runner
from agents.agent import CustomAgent


class CustomRunner(Runner):
    """
    This is the machinery that runs the agent in an environment.
    It is purely related to perform policy inference at each time step
    given the last observations, and feeding the reward signal to the
    appropriate function (feed_return) of the CustomAgent.

    The only difference between this class and its super is that it
    overrides the step method to inform the agent about the end of
    an episode.
    """

    def __init__(self,
                 environment,
                 agent,
                 render=False,
                 verbose=False,
                 vverbose=False,
                 parameters=None,
                 level=None,
                 max_iter=None,
                 log_file_path='runner.log',
                 machine_log_file_path='machine_logs.csv'):

        # Sanity checks.
        assert isinstance(environment, RunEnv)
        assert isinstance(agent, CustomAgent)

        super().__init__(environment,
                         agent,
                         render,
                         verbose,
                         vverbose,
                         parameters,
                         level,
                         max_iter,
                         log_file_path,
                         machine_log_file_path)

    def step(self, observation):
        """
        Performs a full RL step: the agent acts given an observation,
        receives and process the reward, and the env is reset if the
        end of an episode is reached; this also logs the variables of
        the system including actions and observations.

        :param observation: input observation to be given to the agent
        :return: (new observation, action taken, reward received)
        """

        self.logger.debug('observation: ' + str(self.environment.observation_space.array_to_observation(observation)))
        action = self.agent.act(observation)

        # Update the environment with the chosen action
        observation, rewards_list, done, info = self.environment.step(action, do_sum=False)
        if done:
            self.logger.warning('\b\b\bGAME OVER! Resetting grid... (hint: %s)' % info.text)
            observation = self.environment.reset()
        elif info:
            self.logger.warning(info.text)

        reward = sum(rewards_list)

        if self.render:
            self.environment.render()

        self.agent.feed_return(action, observation, rewards_list, done)

        self.logger.debug('action: {}'.format(action))
        self.logger.debug('reward: {}'.format('[' + ','.join(list(map(str, rewards_list))) + ']'))
        self.logger.debug('done: {}'.format(done))
        self.logger.debug('info: {}'.format(info if not info else info.text))

        return observation, action, reward, rewards_list, done

    def loop(self, iterations, episodes=1):
        """
        Runs the simulator for the given number of episodes.

        :param iterations: int of maximum number of iterations per episode
        :param episodes: int of number of episodes, each resetting the environment at the beginning
        :return:
        """

        cumulative_reward = 0.0
        for i_episode in range(episodes):
            cumulative_reward = 0.0
            step = 0
            observation = self.environment.reset()
            while True:
                step += 1
                (observation, action, reward, reward_as_list, done) = self.step(observation)
                cumulative_reward += reward + 5
                self.logger.info("step %d - episode %d - reward: %.2f; " % (step, i_episode, reward + 5))
                self.dump_machinelogs(step, done, reward + 5, reward_as_list, cumulative_reward,
                                      self.environment.get_current_datetime())
                if done:
                    break

                if step > iterations:
                    break
            self.logger.info("ITERATION %d - cumulative reward: %.2f" % (i_episode, cumulative_reward))

        return cumulative_reward
