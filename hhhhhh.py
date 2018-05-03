import os
import os.path as osp
import gym
import gym_gvgai
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from PIL import Image

from model import Model
from runner import Runner

from baselines.a2c.policies import CnnPolicy

from baselines.bench import Monitor
from baselines.common.atari_wrappers import make_atari, NoopResetEnv, MaxAndSkipEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv, FrameStack
from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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

def test(policy, model_steps, env_id, seed, nsteps=5, nstack=1, total_timesteps=1000, num_env=1, runs=100, render=False):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy

    tf.reset_default_graph()
    set_global_seeds(seed)

    env = make_gvgai_env(env_id, num_env, seed)

    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, nenvs=num_env, nsteps=nsteps, nstack=nstack, num_env=num_env)

    try:
        model.load(env_id, model_steps)
    except OSError as e:
        print("Model not found with env " + env_id + " and steps " + str(model_steps))
        env.close()
        return
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
    parser = arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--model-steps', help="Local path to the trained model", type=int, default=20000)
    parser.add_argument('--runs', help='Number of runs to evaluate the model', type=int, default=2)
    parser.add_argument('--num-envs', help='Number of environments/workers to run in parallel', type=int, default=1)
    parser.add_argument('--env', help='environment ID', default='aliens-gvgai-v0')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render screen (default: False)')

    args = parser.parse_args()
    #logger.configure() # Not sure whether this should be called

    # Use args.num_timesteps
    # Use args.env
    #args.env = "aliens-gvgai-v0"

    test(args.policy, args.model_steps, args.env, seed=1, num_env=args.num_envs, runs=args.runs, render=args.render)

if __name__ == '__main__':
    main()
