# import gym
import gymnasium as gym
import minigrid as gym_minigrid
from gymnasium import spaces


class AskAction(gym.Wrapper):
    def __init__(self, env):
        super(AskAction, self).__init__(env)
        old_n = self.action_space.n
        self.action_space = gym.spaces.Discrete(n=old_n+1)


class CartPoleConfig(gym.Wrapper):
    def __init__(self, env, config_type):
        super(CartPoleConfig, self).__init__(env)
        if config_type == 'easy':
            self.unwrapped.length = 0.5
        if config_type == 'medium':
            self.unwrapped.length = 1.0
        if config_type == 'hard':
            self.unwrapped.length = 2.0


class MiniGridAction(gym.Wrapper):
    def __init__(self, env):
        super(MiniGridAction, self).__init__(env)
        old_n = self.action_space.n
        self.action_space = gym.spaces.Discrete(n=old_n-1)


class MiniGridObs(gym.Wrapper):
    def __init__(self, env):
        super(MiniGridObs, self).__init__(env)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed)[0]

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, info


def cartpole_easy_wrapper(env: gym.Env) -> gym.Env:
    env = CartPoleConfig(env, 'easy')
    return env


def cartpole_medium_wrapper(env: gym.Env) -> gym.Env:
    env = CartPoleConfig(env, 'medium')
    return env


def cartpole_hard_wrapper(env: gym.Env) -> gym.Env:
    env = CartPoleConfig(env, 'hard')
    return env


def cartpole_easy_ask_wrapper(env: gym.Env) -> gym.Env:
    env = AskAction(env)
    env = CartPoleConfig(env, 'easy')
    return env


def cartpole_medium_ask_wrapper(env: gym.Env) -> gym.Env:
    env = AskAction(env)
    env = CartPoleConfig(env, 'medium')
    return env


def cartpole_hard_ask_wrapper(env: gym.Env) -> gym.Env:
    env = AskAction(env)
    env = CartPoleConfig(env, 'hard')
    return env


def lunarlander_easy_wrapper(env: gym.Env) -> gym.Env:
    return env


def lunarlander_easy_ask_wrapper(env: gym.Env) -> gym.Env:
    env = AskAction(env)
    return env


def minigrid_wrapper(env: gym.Env) -> gym.Env:
    env = gym_minigrid.wrappers.FlatObsWrapper(env)
    env = MiniGridObs(env)
    env = MiniGridAction(env)
    return env


def minigrid_ask_wrapper(env: gym.Env) -> gym.Env:
    env = gym_minigrid.wrappers.FlatObsWrapper(env)
    env = MiniGridObs(env)
    env = MiniGridAction(env)
    env = AskAction(env)
    return env


def wrap_exp(env_id, mode, level, use_baseline_ask):
    if env_id.split('-')[0] == 'CartPole':
        if mode == 'vanilla' or use_baseline_ask != '':
            if level == 'easy':
                return cartpole_easy_wrapper
            if level == 'medium':
                return cartpole_medium_wrapper
            if level == 'hard':
                return cartpole_hard_wrapper
        if mode == 'human':
            if level == 'easy':
                return cartpole_easy_ask_wrapper
            if level == 'medium':
                return cartpole_medium_ask_wrapper
            if level == 'hard':
                return cartpole_hard_ask_wrapper

    if env_id.split('-')[0] == 'LunarLander':
        if mode == 'vanilla' or use_baseline_ask != '':
            if level == 'easy':
                return lunarlander_easy_wrapper
        if mode == 'human':
            if level == 'easy':
                return lunarlander_easy_ask_wrapper

    if env_id.split('-')[1] == 'MultiRoom':
        if mode == 'vanilla' or use_baseline_ask != '':
            if level == 'easy':
                return minigrid_wrapper, 'MiniGrid-MultiRoom-N2-S4-v0'
        if mode == 'human':
            if level == 'easy':
                return minigrid_ask_wrapper, 'MiniGrid-MultiRoom-N2-S4-v0'

    if env_id.split('-')[1] == 'DoorKey':
        if mode == 'vanilla' or use_baseline_ask != '':
            if level == 'easy':
                return minigrid_wrapper, 'MiniGrid-DoorKey-5x5-v0'
            if level == 'medium':
                return minigrid_wrapper, 'MiniGrid-DoorKey-6x6-v0'
            if level == 'hard':
                return minigrid_wrapper, 'MiniGrid-DoorKey-8x8-v0'
        if mode == 'human':
            if level == 'easy':
                return minigrid_ask_wrapper, 'MiniGrid-DoorKey-5x5-v0'
            if level == 'medium':
                return minigrid_ask_wrapper, 'MiniGrid-DoorKey-6x6-v0'
            if level == 'hard':
                return minigrid_ask_wrapper, 'MiniGrid-DoorKey-8x8-v0'

    assert False, "wrapper not exist"
