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

    mean_score = np.mean(runner.final_rewards[:runs])
    std_score = np.std(runner.final_rewards[:runs])

    print("Mean score=" + str(mean_score))
    print("Games=" + str(runs))
    print("Std. dev. score=" + str(std_score))

    env.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--model-steps', help="Local path to the trained model", type=int, default=7*1000000)
    parser.add_argument('--runs', help='Number of runs to evaluate the model', type=int, default=10)
    parser.add_argument('--num-envs', help='Number of environments/workers to run in parallel', type=int, default=5)
    parser.add_argument('--env', help='environment ID', default='boulderdash')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--model-name', help='Name of the model', default="gvgai-boulderdash-lvl0-v0_lgRandomSelector")
    parser.add_argument('--level', help='Level to test on', default=3)
    parser.add_argument('--level-selector', help='Level selector to use in training',
                        choices=[None, 'random-all', 'random-0123', 'pcg-random', 'pgc-progressive'], default=None)
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render screen (default: False)')

    args = parser.parse_args()

    # Environment name
    env_id = "gvgai-" + args.env + "-lvl" + str(args.level) + "-v0"

    # Level selector
    level_selector = None
    if args.level_selector is not None:
        make_path('./levels/')
        path = os.path.realpath('./levels')
        if args.level_selector == "random-all":
            level_selector = RandomSelector(path, args.env, [0, 1, 2, 3, 4])
        if args.level_selector == "random-0123":
            level_selector = RandomSelector(path, args.env, [0, 1, 2, 3])
        if args.level_selector == "pcg-random":
            level_selector = RandomPCGSelector(path, args.env)
        if args.level_selector == "random":
            level_selector = ProgressivePCGSelector(path, args.env)

    env = make_gvgai_env(env_id, args.num_envs, args.seed, level_selector=level_selector)

    if args.policy == 'cnn':
        policy_fn = CnnPolicy
    elif args.policy == 'lstm':
        policy_fn = LstmPolicy
    elif args.policy == 'lnlstm':
        policy_fn = LnLstmPolicy

    tf.reset_default_graph()

    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, nenvs=args.num_envs, nsteps=5)

    try:
        model.load(args.model_name, args.model_steps)
    except Exception as e:
        print(e)
        env.close()
        return

    eval(model, env, runs=args.runs, render=args.render, level_selector=level_selector)


if __name__ == '__main__':
    main()
