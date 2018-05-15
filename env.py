import os
import gym_gvgai
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from baselines import logger
from baselines.common.atari_wrappers import *
from baselines.common import set_global_seeds
from baselines.bench import Monitor


def worker(remote, parent_remote, env_fn_wrapper, level_selector=None):
    parent_remote.close()
    env = env_fn_wrapper.x()
    level_selector = level_selector
    level = None
    score = 0
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            score += reward
            if done:
                if level_selector is not None:
                    if level_selector.get_game() == "boulderdash":
                        level_selector.report(level, score >= 20)
                    else:
                        level_selector.report(level, False if info['winner'] == 'PLAYER_LOSES' else True)
                    level = level_selector.get_level()
                    env.unwrapped._setLevel(level)
                ob = env.reset()
                score = 0
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            if level_selector is not None:
                level = level_selector.get_level()
                env.unwrapped._setLevel(level)
            ob = env.reset()
            score = 0
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            score = 0
            break
        elif cmd == 'render':
            env.render()
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None, level_selector=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn), level_selector))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self):
        if self.render:
            for remote in self.remotes:
                remote.send(('render', None))

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


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


def make_gvgai_env(env_id, num_env, seed, start_index=0, level_selector=None):
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return wrap_gvgai(env)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)], level_selector=level_selector)


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

