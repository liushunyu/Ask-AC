# This file is here just to define MlpPolicy/CnnPolicy
# that work for AskPPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, ActorCriticAskPolicy, register_policy

AskPolicy = ActorCriticAskPolicy
MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("CnnPolicy", ActorCriticCnnPolicy)
register_policy("AskPolicy", ActorCriticAskPolicy)
