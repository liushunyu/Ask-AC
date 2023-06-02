from stable_baselines3 import PPO, AskPPO, A2C, AskA2C
from stable_baselines3.common.utils import linear_schedule


def get_model(env, env_id, mode, algo, use_baseline_ask, use_ask_loss, ask_threshold, seed, log_dir):
    model = None

    total_timesteps = int(1e4)

    if mode == 'vanilla' or use_baseline_ask != '':
        policy = 'MlpPolicy'
    else:
        policy = 'AskPolicy'

    if env_id.split('-')[0] == 'CartPole' or env_id.split('-')[0] == 'LunarLander':
        if mode == 'vanilla':
            if algo == 'ppo':
                model = PPO(policy, env, learning_rate=linear_schedule(0.001), n_steps=256, batch_size=256, n_epochs=10,
                            gamma=0.99, gae_lambda=0.95, clip_range=linear_schedule(0.2), clip_range_vf=None, ent_coef=0.0,
                            vf_coef=0.5, max_grad_norm=0.5, target_kl=None, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=None, verbose=1, seed=seed)
            if algo == 'a2c':
                model = A2C(policy, env, learning_rate=linear_schedule(0.0007), n_steps=5, gamma=0.99, gae_lambda=1.0, ent_coef=0.0,
                            vf_coef=0.5, max_grad_norm=0.5, rms_prop_eps=1e-05, use_rms_prop=True,
                            normalize_advantage=False, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=None, verbose=1, seed=seed)
        if mode == 'human':
            if algo == 'ppo':
                model = AskPPO(policy, env, learning_rate=linear_schedule(0.001), n_steps=256, batch_size=256, n_epochs=10,
                               gamma=0.99, gae_lambda=0.95, clip_range=linear_schedule(0.2), clip_range_vf=None, ent_coef=0.0,
                               vf_coef=0.5, max_grad_norm=0.5, target_kl=None, tensorboard_log=log_dir,
                               create_eval_env=False, policy_kwargs=None, verbose=1, seed=seed, human_policy=None,
                               use_baseline_ask=use_baseline_ask, ask_threshold=ask_threshold, use_ask_loss=use_ask_loss)
            if algo == 'a2c':
                model = AskA2C(policy, env, learning_rate=linear_schedule(0.0007), n_steps=5, gamma=0.99, gae_lambda=1.0, ent_coef=0.0,
                               vf_coef=0.5, max_grad_norm=0.5, rms_prop_eps=1e-05, use_rms_prop=True,
                               normalize_advantage=False, tensorboard_log=log_dir, create_eval_env=False,
                               policy_kwargs=None, verbose=1, seed=seed, human_policy=None,
                               use_baseline_ask=use_baseline_ask, ask_threshold=ask_threshold, use_ask_loss=use_ask_loss)

    if env_id.split('-')[0] == 'MiniGrid':
        if mode == 'vanilla':
            if algo == 'ppo':
                model = PPO(policy, env, learning_rate=2.5e-4, n_steps=128, batch_size=64, n_epochs=10,
                            gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, ent_coef=0.0,
                            vf_coef=0.5, max_grad_norm=0.5, target_kl=None, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=None, verbose=1, seed=seed)
            if algo == 'a2c':
                model = A2C(policy, env, learning_rate=7e-4, n_steps=5, gamma=0.99,
                            gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, rms_prop_eps=1e-05, use_rms_prop=True,
                            normalize_advantage=False, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=None, verbose=1, seed=seed)
        if mode == 'human':
            if algo == 'ppo':
                model = AskPPO(policy, env, learning_rate=2.5e-4, n_steps=128, batch_size=64, n_epochs=10,
                               gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, ent_coef=0.0,
                               vf_coef=0.5, max_grad_norm=0.5, target_kl=None, tensorboard_log=log_dir,
                               create_eval_env=False, policy_kwargs=None, verbose=1, seed=seed, human_policy=None,
                               use_baseline_ask=use_baseline_ask, ask_threshold=ask_threshold, use_ask_loss=use_ask_loss)
            if algo == 'a2c':
                model = AskA2C(policy, env, learning_rate=7e-4, n_steps=5, gamma=0.99,
                               gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, rms_prop_eps=1e-05, use_rms_prop=True,
                               normalize_advantage=False, tensorboard_log=log_dir, create_eval_env=False,
                               policy_kwargs=None, verbose=1, seed=seed, human_policy=None,
                               use_baseline_ask=use_baseline_ask, ask_threshold=ask_threshold, use_ask_loss=use_ask_loss)

    assert model, "model not exist"
    return model, total_timesteps
