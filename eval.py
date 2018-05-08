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

    env.close()

    return scores


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--runs', help='Number of runs for each model', type=int, default=2)
    parser.add_argument('--num-envs', help='Number of environments/workers to run in parallel', type=int, default=1)
    parser.add_argument('--game', help='Game name (default=zelda)', default='zelda')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--experiment-name', help='Name of the experiment to evaluate, e.g. zelda-ls-pcg-random', default="zelda-lvl-0")
    parser.add_argument('--level', help='Level to test on', default=0)
    parser.add_argument('--selector',
                        help='Level selector to use in test - will ignore the level argument if set (default: None)',
                        choices=[None] + LevelSelector.available, default=None)
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render screen (default: False)')

    args = parser.parse_args()

    # Environment name
    env_id = "gvgai-" + args.game + "-lvl" + str(args.level) + "-v0"

    # Test name
    test_name = args.game
    if args.selector is not None:
        test_name += "-ls-" + args.selector
    else:
        test_name += "-lvl-" + str(args.level)

    # Folders
    score_path = './results/' + args.experiment_name + '/eval/scores/'
    level_path = './results/' + args.experiment_name + '/eval/' + test_name + '/levels/'
    make_path(level_path)
    make_path(score_path)

    # Create file and override if necessary
    score_file = score_path + test_name + ".dat"
    with open(score_file, 'w+') as myfile:
        myfile.write('experiment_id;mean_score;std_score;runs\n')

    # Level selector
    level_selector = LevelSelector.get_selector(args.selector, args.game, level_path)

    mean_scores = []
    std_scores = []
    model_path = './results/' + args.experiment_name + '/models/'
    for model_folder in glob.iglob(model_path + '*/'):

        # Experiment name
        experiment_id = model_folder.split('/')[-2]

        # Find number of steps for last model
        steps = -1
        for model_meta_name in glob.iglob(model_folder + '/*.meta'):
            s = int(model_meta_name.split('.meta')[0].split('/')[-1].split("-")[1])
            if s > steps:
                steps = s

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
            model.load(model_folder, steps)
        except Exception as e:
            print(e)
            env.close()
            return

        scores = eval(model, env, runs=args.runs, render=args.render, level_selector=level_selector)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print("Experiment=" + experiment_id)
        print("Steps trained=" + str(steps))
        print("Eval runs=" + str(args.runs))
        print("Mean score=" + str(mean_score))
        print("Std. dev. score=" + str(std_score))

        # Create log file
        with open(score_file, "a") as myfile:
            line = experiment_id + ";" + str(mean_score) + ';' + str(std_score) + ";" + str(args.runs) + '\n'
            myfile.write(line)

        mean_scores.append(mean_score)
        std_scores.append(std_score)

    # Create log file
    with open(score_file, "a") as myfile:
        line = '\nTotal;' + str(np.mean(mean_scores)) + ";" + str(np.mean(std_scores)) + ";" + str(args.runs * len(mean_scores)) + '\n'
        myfile.write(line)

if __name__ == '__main__':
    main()
