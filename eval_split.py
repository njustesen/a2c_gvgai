import tensorflow as tf
import argparse

from level_selector import *
from model import Model
from runner import Runner
from env import *

from baselines.a2c.utils import make_path
from baselines.a2c.policies import CnnPolicy

from baselines.common import set_global_seeds
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy


def eval(model, env, nsteps=5, runs=100, render=False, level_selector=None):

    runner = Runner(env, model, nsteps=nsteps, gamma=0, render=render)

    while len(runner.final_rewards) < runs:
        obs, states, rewards, masks, actions, values = runner.run()

    scores = runner.final_rewards[:runs]
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    return scores


def test_on(game, level, selector, experiment_name, experiment_id, policy, num_envs=1, seed=0, runs=1000, render=False):

    # Environment name
    env_id = "gvgai-" + game + "-lvl" + str(level) + "-v0"

    # Test name
    test_name = game
    if selector is not None:
        test_name += "-ls-" + selector
    else:
        test_name += "-lvl-" + str(level)

    print("Test name: " + test_name)
    print('Training name: ' + experiment_name)
    print("Training id: " + experiment_id)

    # Level selector
    level_selector = LevelSelector.get_selector(selector, game, level_path)

    env = make_gvgai_env(env_id, num_envs, seed, level_selector=level_selector)

    # Main plots per experiment
    mean_scores = []
    std_scores = []
    model_folder = './results/' + experiment_name + '/models/' + experiment_id + "/"

    # Find number of steps for last model
    steps = -1
    for model_meta_name in glob.iglob(model_folder + '*.meta'):
        s = int(model_meta_name.split('.meta')[0].split('/')[-1].split("-")[1])
        if s > steps:
            steps = s

    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy

    tf.reset_default_graph()

    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, nenvs=num_envs, nsteps=5)

    try:
        model.load(model_folder, steps)
    except Exception as e:
        print(e)
        env.close()
        return

    eval(model, env, runs=runs, render=render, level_selector=level_selector)

    env.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--runs', help='Number of runs for each model', type=int, default=100)
    parser.add_argument('--num-envs', help='Number of environments/workers to run in parallel', type=int, default=10)
    parser.add_argument('--game', help='Game name (default=zelda)', default='zelda')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--experiment-name', help='Name of the experiment to evaluate, e.g. zelda-ls-pcg-random (default=None -> all)', default=None)
    parser.add_argument('--experiment-id', help='Id of the experiment to evaluate')
    parser.add_argument('--level', help='Level (integer) to train on', type=int, default=0)
    parser.add_argument('--selector',
                        help='Level selector to use in training - will ignore the level argument if set (default: None)',
                        choices=[None] + LevelSelector.available, default=None)
    parser.add_argument('--render', action='store_true',
                        help='Render screen (default: False)')

    args = parser.parse_args()

    test_on(args.game, args.level, args.selector, experiment_name=args.experiment_name, experiment_id=args.experiment_id, policy=args.policy, runs=args.runs, seed=args.seed, num_envs=args.num_envs, render=args.render)


if __name__ == '__main__':
    main()
