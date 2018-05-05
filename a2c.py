import time
import tensorflow as tf
import argparse

from model import Model
from runner import Runner
from env import *
from level_selector import *

from baselines.a2c.utils import make_path
from baselines.a2c.policies import CnnPolicy

from baselines.common import set_global_seeds
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy


def learn(policy, env, seed, game_name, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, save_interval=25000, num_env=1, frame_skip=False, level=None, level_selector=None):
    tf.reset_default_graph()
    set_global_seeds(seed)

    log_path = "./logs/a2c/"
    make_path(log_path)
    experiment_name = game_name
    if level is not None:
        experiment_name += "_lvl" + str(level)
    if level_selector is not None:
        experiment_name += "_lg" + level_selector.__class__.__name__
    log_file = log_path + experiment_name + ".log"

    with open(log_file, "a") as myfile:
        line = "episodes; steps; frames; mean_score; std_score; min_score; max_score\n"
        myfile.write(line)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    #num_procs = len(env.remotes) # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    episodes = 0
    next_model_save = save_interval
    model.save(experiment_name, 0)
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run()

        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart
        f = 4 if frame_skip else 1
        fps = int((update*nbatch*f)/nseconds)
        steps = update*nbatch
        frames = steps*f

        # If n final resuts were reported - save the average and std dev
        if len(runner.final_rewards) >= runner.nenv:

            episodes += runner.nenv

            # Extract and remove a number of final rewards equal to the number of workers
            final_rewards = runner.final_rewards[:runner.nenv]
            mean_score = np.mean(final_rewards)
            std_score = np.std(final_rewards)
            min_score = np.min(final_rewards)
            max_score = np.max(final_rewards)
            runner.final_rewards = runner.final_rewards[runner.nenv:]
            logger.record_tabular("mean_score", mean_score)
            logger.record_tabular("std_score", std_score)
            logger.record_tabular("min_score", min_score)
            logger.record_tabular("max_score", max_score)
            logger.record_tabular("steps", steps)
            logger.record_tabular("frames", frames)
            logger.record_tabular("episodes", episodes)
            logger.record_tabular("fps", fps)
            logger.dump_tabular()

            with open(log_file, "a") as myfile:
                if level_selector is not None and isinstance(level_selector, ProgressivePCGSelector):
                    line = str(episodes) + ";" + str(steps) + ";" + str(frames) + ";" + str(mean_score) + ";" + str(
                        std_score) + ";" + str(min_score) + ";" + str(max_score) + ";" + str(level_selector.alpha) + "\n"
                else:
                    line = str(episodes) + ";" + str(steps) + ";" + str(frames) + ";" + str(mean_score) + ";" + str(std_score) + ";" + str(min_score) + ";" + str(max_score) + "\n"
                myfile.write(line)

        if steps >= next_model_save:
            model.save(experiment_name, next_model_save)
            next_model_save += save_interval

    env.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-envs', help='Number of environments/workers to run in parallel', type=int, default=2)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--env', help='Environment ID', default='aliens')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--save-interval', help='Model saving interval in steps', type=int, default=1000000)
    parser.add_argument('--level', help='Level to train on', type=int, default=0)
    parser.add_argument('--level-selector', help='Level selector to use in training', choices=[None, 'random-all', 'random-0123', 'pcg-random', 'pgc-progressive'], default=None)
    args = parser.parse_args()

    # Environment name
    env_id = "gvgai-" + args.env

    # Fixed level?
    env_id += "-lvl" + str(args.level)
    env_id += "-v0"

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

    # Specify model
    if args.policy == 'cnn':
        policy_fn = CnnPolicy
    elif args.policy == 'lstm':
        policy_fn = LstmPolicy
    elif args.policy == 'lnlstm':
        policy_fn = LnLstmPolicy

    learn(policy_fn, env, args.seed, game_name=env_id, total_timesteps=args.num_timesteps, lrschedule=args.lrschedule,
          num_env=args.num_envs, nstack=1, frame_skip=False, save_interval=args.save_interval, level_selector=level_selector)

    env.close()

if __name__ == '__main__':
    main()
