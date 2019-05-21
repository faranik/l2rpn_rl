import pypownet.environment
import numpy as np


def observation_to_state(observation: pypownet.environment.Observation) -> int:
    """
    Given the observations on environment, deduct the state. In this
    implementation all lines with a charge level greater than 70%
    is considered as 1 and others as 0.

    :param observation: Observations of the environment.
    :return: The state as an integer.
    """

    assert isinstance(observation, pypownet.environment.Observation)
    lines_usage = observation.get_lines_capacity_usage()
    binary_lines_usage = np.greater(lines_usage, 1.0)

    state = 0
    for bit in binary_lines_usage:
        state = (state << 1) | bit

    return state


def agents_action_to_envs_action(action, agents_action):
    """
    Transform an agent's action to an environment's action.

    :param action: Instance fo pypownet.game.Action that the
        environment can understand.
    :param agents_action: Integer, representing the index of the line
        extremity that will switch to another substation.
    :return: An instance of pypownet.game.Action
    """

    act = np.zeros(len(action.get_node_splitting_subaction()), np.int)
    act[agents_action] = 1

    action.set_node_splitting_subaction(act)

    return action

