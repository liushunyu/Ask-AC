# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, ActorCriticGnnPolicy, register_policy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
GnnPolicy = ActorCriticGnnPolicy

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("CnnPolicy", ActorCriticCnnPolicy)
register_policy("GnnPolicy", ActorCriticGnnPolicy)
