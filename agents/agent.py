import pypownet.environment
import pypownet.agent as agent

# Examples of baselines agents
import numpy as np


class PolicyIteration(agent.Agent):
    """
    An example of a baseline controller that produce 1 random activation (ie an array with all 0 but one 1).
    """

    def __init__(self, environment):
        super().__init__(environment)

    def act(self, observation):
        action = self.environment.action_space.get_do_nothing_action()
        # # or
        # action_length = self.environment.action_space.n
        # action = np.zeros(action_length)
        action[np.random.randint(action.shape[0])] = 1
        return action
