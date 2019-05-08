import argparse

from pypownet.environment import RunEnv
from pypownet.runner import Runner
import pypownet.agent

parser = argparse.ArgumentParser(description='CLI tool to run experiments using PyPowNet.')
parser.add_argument('-a', '--agent', metavar='AGENT_CLASS', default='DoNothing', type=str,
                    help='class to use for the agent (must be within the \'pypownet/agent.py\' file); '
                         'default class Agent')
parser.add_argument('-n', '--niter', type=int, metavar='NUMBER_ITERATIONS', default='1000',
                    help='number of iterations to simulate (default 1000)')
parser.add_argument('-p', '--parameters', metavar='PARAMETERS_FOLDER', default='./parameters/default14/', type=str,
                    help='parent folder containing the parameters of the simulator to be used (folder should contain '
                         'configuration.json and reference_grid.m)')
parser.add_argument('-lv', '--level', metavar='GAME_LEVEL', type=str, default='level0',
                    help='game level of the timestep entries to be played (default \'level0\')')
parser.add_argument('-s', '--start-id', metavar='CHRONIC_START_ID', type=int, default=0,
                    help='id of the first chronic to be played (default 0)')
parser.add_argument('-lm', '--loop-mode', metavar='CHRONIC_LOOP_MODE', type=str, default='natural',
                    help='the way the game will loop through chronics of the specified game level: "natural" will'
                         ' play chronic in alphabetical order, "random" will load random chronics ids and "fixed"'
                         ' will always play the same chronic folder (default "natural")')
parser.add_argument('-m', '--game-over-mode', metavar='GAME_OVER_MODE', type=str, default='soft',
                    help='game over mode to be played: either "soft", and after each game over the simulator will load '
                         'the next timestep of the same chronic; or "hard", and after each game over the simulator '
                         'will load the first timestep of the next grid, depending on --loop-mode parameter (default '
                         '"soft")')
parser.add_argument('--no-overflow-cutoff', action='store_true',
                    help='disable the grid automatic cut-off of overflows lines (soft and hard) independently from the '
                         'parameters environment used')
parser.add_argument('-r', '--render', action='store_true',
                    help='render the power network observation at each timestep (not available if --batch is not 1)')
parser.add_argument('-la', '--latency', type=float, default=None,
                    help='time to sleep after each frame plot of the renderer (in seconds); note: there are multiple'
                         ' frame plots per timestep (at least 2, varies)')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='display live info of the current experiment including reward, cumulative reward')
parser.add_argument('-vv', '--vverbose', action='store_true',
                    help='display live info + observations and actions played')


def main():
    args = parser.parse_args()
    env_class = RunEnv
    agent_class = eval('pypownet.agent.{}'.format(args.agent))

    # Instantiate environment and agent
    env = env_class(parameters_folder=args.parameters, game_level=args.level,
                    chronic_looping_mode=args.loop_mode, start_id=args.start_id,
                    game_over_mode=args.game_over_mode, renderer_latency=args.latency,
                    without_overflow_cutoff=args.no_overflow_cutoff)
    agent = agent_class(env)
    # Instantiate game runner and loop
    runner = Runner(env, agent, args.render, args.verbose, args.vverbose, args.parameters, args.level, args.niter)
    final_reward = runner.loop(iterations=args.niter)
    print("Obtained a final reward of {}".format(final_reward))


if __name__ == "__main__":
    main()
