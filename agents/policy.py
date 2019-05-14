
class Policy:
    def __init__(self):
        """ Initialize with random values """

    def get_best_action(self, state: int) -> int:
        """
        Get the best action for this state
        :param state: Each bit in state variable represents the state
                      of a line.
        :return: Each bit in returned value is the state of a
                 substation.
        """

    def improve(self) -> "Policy":
        """
        Returns another instance of Policy class which is at least
        as good as this one.
        :return: An instance of Policy class.
        """

    def is_mature(self) -> bool:
        """
        Given the improvements history, is this policy mature?
        I.e. is this policy the optimal policy for this MDP?
        :return: True if we believe the policy converged to the optimal
                 policy for the MDP on which it was trained.
        """
        return False
