# import gym
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from utils.wrappers import wrap_exp


def get_env(env_id, mode, level, seed, use_baseline_ask, algo=None):
    env = None
    use_human_advisor = True
    n_envs = 1 if use_human_advisor else 8
    if env_id.split('-')[0] == 'CartPole' or env_id.split('-')[0] == 'LunarLander':
        wrapper_class = wrap_exp(env_id, mode, level, use_baseline_ask)
        env = make_vec_env(env_id, n_envs=n_envs, seed=seed, wrapper_class=wrapper_class)

    if env_id.split('-')[0] == 'MiniGrid':
        wrapper_class, env_id = wrap_exp(env_id, mode, level, use_baseline_ask)
        env = make_vec_env(env_id, n_envs=n_envs, seed=seed, wrapper_class=wrapper_class, env_kwargs={"render_mode": "human", "agent_pov": True})
        env = VecNormalize(env)

    assert env, "env not exist"

    return env


def get_test_env(env_id, mode, level, seed=0):
    if env_id.split('-')[0] == 'CartPole' or env_id.split('-')[0] == 'LunarLander':
        env = gym.make(env_id)
        env.seed(seed)
        env.action_space.seed(seed)
        env = Monitor(env)
        wrapper_class = wrap_exp(env_id, mode, level, use_baseline_ask='')
        env = wrapper_class(env)

    if env_id.split('-')[0] == 'MiniGrid':
        wrapper_class, env_id = wrap_exp(env_id, mode, level, use_baseline_ask='')
        env = make_vec_env(env_id, n_envs=1, seed=seed, wrapper_class=wrapper_class)
        env = VecNormalize(env)

    assert env, "env not exist"

    return env
