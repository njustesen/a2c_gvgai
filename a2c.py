import time
import tensorflow as tf
import argparse
import glob
import random

from model import Model
from runner import Runner
from env import *
from level_selector import *


import uuid

from baselines.a2c.utils import make_path
from baselines.a2c.policies import CnnPolicy

from baselines.common import set_global_seeds
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy


def learn(policy, env, experiment_name, experiment_id, seed=None, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, save_interval=25000, frame_skip=False, level_selector=None, render=False):
    
    if seed is None:
        seed = random.randint(0, 10000000)
    
    tf.reset_default_graph()
    set_global_seeds(seed)

    # Check if experiment_id exists - then continue
    model_path = './results/' + experiment_name + '/models/'
    steps = 0
    resume = False
    for model_folder in glob.iglob(model_path + '*/'):

        # Experiment id
        id = model_folder.split('/')[-2]

        if id != experiment_id:
            continue

        # Find number of steps for last model
        for model_meta_name in glob.iglob(model_folder + '/*.meta'):
            resume = True
            s = int(model_meta_name.split('.meta')[0].split('/')[-1].split("-")[1])
            if s >= steps:
                steps = s

    # Log file path
    log_path = "./results/" + experiment_name + "/logs/"
    make_path(log_path)
    log_file = log_path + experiment_id + ".log"

    f = open("demofile.txt", "w")

    # Create log file
    if not resume:
        with open(log_file, "a") as myfile:
            line = "episodes; steps; frames; mean_score; std_score; min_score; max_score; difficulty; policy_loss; value_loss; episode_length;fps;\n"
            myfile.write(line)

    # Model folder path
    model_path = "./results/" + experiment_name + "/models/" + experiment_id + "/"
    make_path(model_path)

    # Create model
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    #num_procs = len(env.remotes) # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)

    if resume:
        try:
            model.load(model_path, steps)
        except Exception as e:
            print(e)
            env.close()
            return

    # Create parallel runner
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma, render=render)

    # Training loop
    nbatch = nenvs*nsteps
    tstart = time.time()
    next_model_save = steps + save_interval
    start_update = steps//nbatch
    episodes = 0

    if not resume:
        model.save(model_path, 0)
    else:
        with open(log_file, "r") as myfile:
            for line in myfile:
                pass
            episodes = int(line.split(';')[0])
            if level_selector is not None and line.split(';')[-1] != '' and hasattr(level_selector, 'difficulty'):
                print("Restoring difficulty to " + line.split(';')[-1])
                level_selector.difficulty = int(line.split(';')[-1])

    last_frames = 0
    for update in range(start_update, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run()

        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart
        f = 4 if frame_skip else 1
        fps = int((update*nbatch*f)/nseconds)
        steps = update*nbatch
        frames = steps*f    # Frames is the same as steps if no frame skipping - both are logged

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

            # Debug logging
            f = frames - last_frames
            last_frames = frames
            frames_per_episode = runner.nenv / f
            last_level_id = runner.last_level_id

            # Log using baselines logger
            logger.record_tabular("mean_score", mean_score)
            logger.record_tabular("std_score", std_score)
            logger.record_tabular("min_score", min_score)
            logger.record_tabular("max_score", max_score)
            logger.record_tabular("steps", steps)
            logger.record_tabular("frames", frames)
            logger.record_tabular("episodes", episodes)
            logger.record_tabular("fps", fps)
            if level_selector is not None:
                logger.record_tabular("difficulty", str(level_selector.get_info()))
            logger.dump_tabular()

            # Log to file
            with open(log_file, "a") as myfile:
                dif = ""
                if level_selector is not None:
                    dif = str(level_selector.get_info())
                line = str(episodes) + ";" + str(steps) + ";" + str(frames) + ";" + str(mean_score) + ";" + str(std_score) + ";" + str(min_score) + ";" + str(max_score) + ";" + dif + ";" + str(policy_loss) + ";" + str(value_loss) + ";" + str(frames_per_episode) + ";" + str(fps) + ";" + "\n"
                myfile.write(line)

        # Save model
        if steps >= next_model_save:
            model.save(model_path, next_model_save)
            next_model_save += save_interval

    # Save model in the end
    model.save(model_path, total_timesteps)

    env.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-envs', help='Number of environments/workers to run in parallel (default=12)', type=int, default=12)
    parser.add_argument('--num-timesteps', help='Number of timesteps to train the model', type=int, default=int(20e6))
    parser.add_argument('--game', help='Game name (default=zelda)', default='zelda')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--save-interval', help='Model saving interval in steps', type=int, default=int(1e6))
    parser.add_argument('--level', help='Level (integer) to train on', type=int, default=0)
    parser.add_argument('--resume', help='The experiment id to resume', default=None)
    parser.add_argument('--repetitions', help='Number of repetitions to run sequentially (default=1)', type=int, default=1)
    parser.add_argument('--selector', help='Level selector to use in training - will ignore the level argument if set (default: None)',
                        choices=[None] + LevelSelector.available, default=None)
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render screen (default: False)')
    args = parser.parse_args()

    # Gym environment name
    env_id = "gvgai-" + args.game + "-lvl" + str(args.level) + "-v0"

    # Experiment name
    make_path("./results")
    experiment_name = args.game
    if args.selector is not None:
        experiment_name += "-ls-" + args.selector
    else:
        experiment_name += "-lvl-" + str(args.level)
    make_path("./results/" + experiment_name)

    for i in range(args.repetitions):

        print("Starting experiment " + str(i+1) + " of " + str(args.repetitions))

        # Unique id for experiment
        if args.resume is None:
            experiment_id = str(uuid.uuid1())
        else:
            experiment_id = args.resume

        # Level selector
        level_path = './results/' + experiment_name + '/levels/' + experiment_id + '/'
        level_selector = LevelSelector.get_selector(args.selector, args.game, level_path)

        # Make gym environment
        env = make_gvgai_env(env_id=env_id,
                             num_env=args.num_envs,
                             seed=args.seed,
                             level_selector=level_selector)

        # Atari
        #env_id = "BreakoutNoFrameskip-v4"
        #env = make_atari_env(env_id, args.num_envs, args.seed)

        # Select model
        if args.policy == 'cnn':
            policy_fn = CnnPolicy
        elif args.policy == 'lstm':
            policy_fn = LstmPolicy
        elif args.policy == 'lnlstm':
            policy_fn = LnLstmPolicy

        learn(policy=policy_fn,
              env=env,
              experiment_name=experiment_name,
              experiment_id=experiment_id,
              seed=args.seed,
              total_timesteps=args.num_timesteps,
              lrschedule=args.lrschedule,
              frame_skip=False,
              save_interval=args.save_interval,
              level_selector=level_selector,
              render=args.render)

        env.close()

        print("Experiment DONE")


if __name__ == '__main__':
    main()
