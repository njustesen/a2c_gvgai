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

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.policies import CnnPolicy
from baselines.a2c.utils import cat_entropy, mse

from baselines.bench import Monitor
from baselines.common.atari_wrappers import make_atari, NoopResetEnv, MaxAndSkipEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv, FrameStack
from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import tf_util

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
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

def wrap_gvgai(env, frame_stack=False, scale=False, clip_rewards=False, noop_reset=False, frame_skip=False):
    """Configure environment for DeepMind-style Atari.
    """
    env = WarpFrame(env)
    if frame_skip:
        env = MaxAndSkipEnv(env, skip=4)
    if noop_reset:
        env = NoopResetEnv(env, noop_max=30)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_env,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_env,
                                inter_op_parallelism_threads=num_env)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(path, model_name):
            ps = sess.run(params)
            make_path(path)
            joblib.dump(ps, path + model_name)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        self.nenv = env.num_envs
        self.batch_ob_shape = (self.nenv*nsteps, nh, nw, nc)
        self.obs = np.zeros((self.nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(self.nenv)]
        self.final_rewards = []
        self.episode_rewards = [0 for _ in range(self.nenv)]

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            # Iterate rewards step-by-step and add to final scores if done
            for i in range(self.nsteps):
                # Add reward to episode reward
                self.episode_rewards[n] += rewards[i]
                if dones[i] == 1:
                    # Add final result to episode rewards
                    self.final_rewards.append(self.episode_rewards[n])
                    # Reset local episode reward
                    self.episode_rewards[n] = 0
            # Discount rewards
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

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
    next_model_save = 0
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
            next_model_save += save_interval
            model.save("./models/a2c/" + game_name + "/", "model_episodes" + str(episodes) + "_frames" + str(frames) + ".pkl")

    env.close()

def enjoy(policy, model_path, env_id, seed, nsteps=5, nstack=4, total_timesteps=1000):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy

    tf.reset_default_graph()
    set_global_seeds(seed)

    env = VecFrameStack(make_atari_env(env_id, 1, seed), 4)
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, nenvs=1, nsteps=nsteps, nstack=nstack, num_env=1)
    model.load(model_path)
    runner = Runner(env, model, nsteps=nsteps, gamma=0)

    final_rewards = []
    current_rewards = 0
    for update in range(1, total_timesteps):
        obs, states, rewards, masks, actions, values = runner.run()

    print("Mean score="+ str(np.mean(runner.final_rewards)))
    print("Games="+str(len(runner.final_rewards)))
    print("Std. dev. score="+ str(np.std(runner.final_rewards)))

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
            return wrap_gvgai(env, scale=False, frame_stack=False, clip_rewards=False, noop_reset=False, frame_skip=frame_skip)
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
    parser.add_argument('--num-envs', help='Number of environments/workers to run in parallel', type=int, default=12)
    parser.add_argument('--num-timesteps', type=int, default=int(10e7))
    #parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--env', help='environment ID', default='aliens-gvgai-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--save-interval', help='Model saving interval in steps', type=int, default=25000)
    args = parser.parse_args()
    #logger.configure() # Not sure whether this should be called

    # Use args.num_timesteps
    # Use args.env
    #args.env = "aliens-gvgai-v0"

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy, lrschedule=args.lrschedule, num_env=args.num_envs, save_interval=args.save_interval, frame_skip=False)
    #enjoy(args.policy, "./models/a2c/" + args.env + "/model_episodes0_steps10.pkl", args.env, seed=1)

if __name__ == '__main__':
    main()
