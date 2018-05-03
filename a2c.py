import os
import gym
import gym_gvgai
import time
import numpy as np
import tensorflow as tf
from baselines import logger

from PIL import Image

from model import Model
from runner import Runner

from baselines.a2c.utils import make_path
from baselines.a2c.policies import CnnPolicy
from baselines.bench import Monitor
from baselines.common.atari_wrappers import make_atari, NoopResetEnv, MaxAndSkipEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv, FrameStack
from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def wrap_gvgai(env, frame_stack=False, scale=False, clip_rewards=False, noop_reset=False, frame_skip=False, scale_float=False):
    """Configure environment for DeepMind-style Atari.
    """
    if scale_float:
        env = ScaledFloatFrame(env)
    if scale:
        env = WarpFrame(env)
    if frame_skip:
        env = MaxAndSkipEnv(env, skip=4)
    if noop_reset:
        env = NoopResetEnv(env, noop_max=30)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

def learn(policy, env, seed, game_name, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, save_interval=25000, num_env=1, frame_skip=False):
    tf.reset_default_graph()
    set_global_seeds(seed)

    log_path = "./logs/a2c/"
    make_path(log_path)
    log_file = log_path + game_name + ".log"

    with open(log_file, "a") as myfile:
        line = "episodes; steps; frames; mean_score; std_score; min_score; max_score\n"
        myfile.write(line)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    #num_procs = len(env.remotes) # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_env=num_env, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    episodes = 0
    next_model_save = save_interval
    model.save(game_name, 0)
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
                line = str(episodes) + ";" + str(steps) + ";" + str(frames) + ";" + str(mean_score) + ";" + str(std_score) + ";" + str(min_score) + ";" + str(max_score) + "\n"
                myfile.write(line)

        if steps >= next_model_save:
            model.save(game_name, next_model_save)
            next_model_save += save_interval

    env.close()

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    return parser

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_gvgai_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, frame_skip=False):
    #if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return wrap_gvgai(env)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env, save_interval, frame_skip):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy

    # Atari
    #env = make_atari_env(env_id, num_env, seed)
    #env = VecFrameStack(env, 4)

    # GVG-AI
    env = make_gvgai_env(env_id, num_env, seed)
    #env = VecFrameStack(env, 4)

    learn(policy_fn, env, seed, game_name=env_id, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, num_env=num_env, nstack=1, frame_skip=frame_skip, save_interval=save_interval)
    env.close()

def main():
    parser = arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-envs', help='Number of environments/workers to run in parallel', type=int, default=2)
    parser.add_argument('--num-timesteps', type=int, default=int(100e6))
    #parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--env', help='environment ID', default='aliens-gvgai-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--save-interval', help='Model saving interval in steps', type=int, default=1000)
    args = parser.parse_args()
    #logger.configure() # Not sure whether this should be called

    # Use args.num_timesteps
    # Use args.env
    #args.env = "aliens-gvgai-v0"

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy, lrschedule=args.lrschedule, num_env=args.num_envs, save_interval=args.save_interval, frame_skip=False)
    #evaluate(args.policy, "./models/a2c/" + args.env + "/model_episodes0_steps10.pkl", args.env, seed=1)

if __name__ == '__main__':
    main()
