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


def test_on(game, level, selector, experiment, policy, num_envs=1, seed=0, runs=100, render=False):

    # Environment name
    env_id = "gvgai-" + game + "-lvl" + str(level) + "-v0"

    # Test name
    test_name = game
    if selector is not None:
        test_name += "-ls-" + selector
    else:
        test_name += "-lvl-" + str(level)

    experiments = []
    if experiment is not None:
        experiments.append(experiment)
    else:
        for experiment_folder in glob.iglob('./results/*/'):
            name = experiment_folder.split('/')[-2]
            experiments.append(name)

    for experiment_name in experiments:

        print('Starting ' + experiment_name)

        # Folders
        score_path = './results/' + experiment_name + '/eval/' + test_name + '/scores/'
        level_path = './results/' + experiment_name + '/eval/' + test_name + '/levels/'
        make_path(level_path)
        make_path(score_path)

        # Create file and override if necessary
        score_file = score_path + test_name + ".dat"
        with open(score_file, 'w+') as myfile:
            myfile.write('')

        # Level selector
        level_selector = LevelSelector.get_selector(selector, game, level_path)

        # Main plots per experiment
        mean_scores = []
        std_scores = []
        model_path = './results/' + experiment_name + '/models/'
        for model_folder in glob.iglob(model_path + '*/'):

            # Experiment name
            experiment_id = model_folder.split('/')[-2]

            # Find number of steps for last model
            steps = -1
            for model_meta_name in glob.iglob(model_folder + '/*.meta'):
                s = int(model_meta_name.split('.meta')[0].split('/')[-1].split("-")[1])
                if s > steps:
                    steps = s

            env = make_gvgai_env(env_id, num_envs, seed, level_selector=level_selector)

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

            scores = eval(model, env, runs=runs, render=render, level_selector=level_selector)

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print("Testing on=" + test_name)
            print("Trained on=" + experiment_name)
            print("Model id=" + experiment_id)
            print("Steps trained=" + str(steps))
            print("Runs=" + str(runs))
            print("Mean score=" + str(mean_score))
            print("Std. dev.=" + str(std_score))

            # Save results
            with open(score_file, "a") as myfile:
                line = "Testing on=" + test_name + "\n"
                line += "Trained on=" + experiment_name + "\n"
                line += "Id=" + experiment_id + "\n"
                line += "Steps trained=" + str(steps) + "\n"
                line += "Runs=" + str(runs) + "\n"
                line += "Mean score=" + str(mean_score) + "\n"
                line += "Std. dev.=" + str(std_score) + "\n"
                line += "\n"
                myfile.write(line)

            mean_scores.append(mean_score)
            std_scores.append(std_score)

            env.close()

        # Create log file
        with open(score_file, "a") as myfile:
            line = 'Total' + '\n'
            line += 'Mean score=' + str(np.mean(mean_scores)) + '\n'
            line += 'Mean std. devs=' + str(np.mean(std_scores)) + '\n'
            line += 'Runs=' + str(runs * len(mean_scores)) + '\n'
            myfile.write(line)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--runs', help='Number of runs for each model', type=int, default=2)
    parser.add_argument('--num-envs', help='Number of environments/workers to run in parallel', type=int, default=2)
    parser.add_argument('--game', help='Game name (default=zelda)', default='zelda')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--experiment-name', help='Name of the experiment to evaluate, e.g. zelda-ls-pcg-random (default=None -> all)', default=None)
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render screen (default: False)')

    args = parser.parse_args()

    levels = [0, 1, 2, 3, 4]
    selectors = ['seq-0',
                 'seq-3',
                 'seq-5',
                 'seq-7',
                 'seq-10']

    for level in levels:
        test_on(args.game, level, None, experiment=args.experiment_name, policy=args.policy, runs=args.runs, seed=args.seed)

    for selector in selectors:
        test_on(args.game, 0, selector, experiment=args.experiment_name, policy=args.policy, runs=args.runs, seed=args.seed)


if __name__ == '__main__':
    main()
