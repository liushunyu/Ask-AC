from typing import Any, Callable, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.on_ask_policy_algorithm import OnAskPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticAskPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance


class AskA2C(OnAskPolicyAlgorithm):
    """
    Ask Advantage Actor Critic (Ask A2C)

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticAskPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        human_policy: Callable = None,
        use_ask_loss: bool = True,
        use_baseline_ask: str = '',
        ask_threshold: float = 0.2,
        _init_setup_model: bool = True,
    ):

        if use_baseline_ask != '':
            assert use_baseline_ask in ['cm', 'diff', 'var'], 'use_baseline_ask not exist'

        if use_baseline_ask != '':
            use_ask_loss = False

        super(AskA2C, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            human_policy=human_policy,
            use_ask_loss=use_ask_loss,
            use_baseline_ask=use_baseline_ask,
            ask_threshold=ask_threshold,
            _init_setup_model=False,
        )

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        self.momentum_vl = 0.0

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # ===================================== ask loss start =====================================

        ask_observations_nums = []
        ask_losses = []

        if self.use_ask_loss:
            def get_ask_observations(rollout_data):
                with th.no_grad():
                    _, latent_vf = self.policy._get_latent(rollout_data.observations)
                    values_pred = self.policy.value_net(latent_vf).flatten()

                    # # ================ threshold ask loss start ================
                    # ask_threshold = 3000  # CartPole 3000, DoorKey 3
                    # ask_observations = rollout_data.observations[((rollout_data.returns - values_pred) ** 2) > ask_threshold]
                    # # ================ threshold ask loss end ================

                    # ================ adaptive ask loss start ================
                    current_vl = ((rollout_data.returns - values_pred) ** 2).detach().cpu().numpy()
                    self.momentum_vl = 0.9 * self.momentum_vl + 0.1 * current_vl.mean()

                    unstable_rate = (current_vl.mean() * (1 - 0.9) / self.momentum_vl) * 0.1
                    unstable_num = int(np.ceil(unstable_rate * len(rollout_data.observations)))

                    vl_rank = np.argsort(current_vl)[::-1]
                    ask_observations = rollout_data.observations[list(vl_rank[0:unstable_num]), :]
                    # ================ adaptive ask loss end ================

                    return ask_observations

            for epoch in range(1):
                for rollout_data in self.rollout_buffer.get(batch_size=None):
                    al_coef = 0.5
                    ask_loss = th.as_tensor(0.0)
                    ask_observations = get_ask_observations(rollout_data)
                    ask_observations_num = len(ask_observations)
                    ask_observations_nums.append(ask_observations_num)

                    if ask_observations_num > 0:
                        al_coef = al_coef * ask_observations_num / len(rollout_data.observations)
                        latent_pi, _ = self.policy._get_latent(ask_observations)
                        ask_distribution, standard_distribution = self.policy._get_action_dist_from_latent(latent_pi)
                        batch_size = latent_pi.shape[0]
                        probs = th.zeros((batch_size, self.action_space.n), dtype=th.float32).to(self.device)
                        probs[:, 0] = ask_distribution.distribution.probs[:, 0]
                        probs[:, 1:] = ask_distribution.distribution.probs[:, 1].unsqueeze(1) * standard_distribution.distribution.probs
                        ask_actions = th.as_tensor([0] * ask_observations_num).long().to(self.device)
                        ask_loss = F.nll_loss(probs, ask_actions)

                    ask_losses.append(ask_loss.item())

                    if ask_loss.item() != 0.0:
                        loss = al_coef * ask_loss
                        self.policy.optimizer.zero_grad()
                        loss.backward()
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.policy.optimizer.step()

        # ===================================== ask loss end =====================================

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # TODO: avoid second computation of everything because of the gradient
            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # ===================================== human loss start =====================================

        human_actions_nums = []
        human_losses = []

        for epoch in range(1):
            for rollout_data in self.rollout_buffer.get(batch_size=None):
                human_actions = rollout_data.human_actions
                if isinstance(self.action_space, spaces.Discrete):
                    human_actions = rollout_data.human_actions.long().flatten()

                hl_coef = 1
                human_loss = th.as_tensor(0.0)
                human_actions_num = int((human_actions != -1).sum().cpu().numpy())
                human_actions_nums.append(human_actions_num)

                if human_actions_num > 0:
                    hl_coef = hl_coef * human_actions_num / len(rollout_data.observations)

                    if self.use_baseline_ask != '':
                        latent_pi, _, latent_sde = self.policy._get_latent(rollout_data.observations)
                        distribution = self.policy._get_action_dist_from_latent(latent_pi, latent_sde)
                        probs = distribution.distribution.probs
                    else:
                        latent_pi, _ = self.policy._get_latent(rollout_data.observations)
                        ask_distribution, standard_distribution = self.policy._get_action_dist_from_latent(latent_pi)
                        batch_size = latent_pi.shape[0]
                        probs = th.zeros((batch_size, self.action_space.n), dtype=th.float32).to(self.device)
                        probs[:, 0] = ask_distribution.distribution.probs[:, 0]
                        probs[:, 1:] = ask_distribution.distribution.probs[:, 1].unsqueeze(1) * standard_distribution.distribution.probs

                    human_loss = F.nll_loss(probs[human_actions != -1], human_actions[human_actions != -1])

                human_losses.append(human_loss.item())

                if human_loss.item() != 0.0:
                    loss = hl_coef * human_loss
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

        # ===================================== human loss end =====================================

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/explained_variance", explained_var)
        logger.record("train/entropy_loss", entropy_loss.item())
        logger.record("train/policy_loss", policy_loss.item())
        logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        # ===================================== Ask-AC log start =====================================
        if self.use_ask_loss:
            logger.record("train/ask_loss", np.mean(ask_losses))
            logger.record("train/ask_observations_num", np.mean(ask_observations_nums))

        logger.record("train/human_loss", np.mean(human_losses))
        logger.record("train/human_actions_num", np.mean(human_actions_nums))
        # ===================================== Ask-AC log end =====================================

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "AskA2C",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "AskA2C":

        return super(AskA2C, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
