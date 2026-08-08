"""Evaluation callback for SBX / SB3 algorithms.

For off-policy algorithms, evaluates at a fixed episode interval matching the
``log_interval`` semantics (which count episodes, not timesteps).
For on-policy algorithms, evaluates at the end of every ``eval_freq`` rollouts.
"""

from __future__ import annotations

import gymnasium as gym
import jax
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def evaluate_policy(model, env, n_episodes, deterministic=True, gamma=0.99):
    """Run *n_episodes* and return per-episode undiscounted and discounted returns."""
    episode_returns = []
    episode_discounted_returns = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_discounted = 0.0
        gamma_power = 1.0
        length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
            ep_discounted += gamma_power * float(reward)
            gamma_power *= gamma
            length += 1

        episode_returns.append(ep_return)
        episode_discounted_returns.append(ep_discounted)
        episode_lengths.append(length)

    return episode_returns, episode_discounted_returns, episode_lengths


class EvalCallback(BaseCallback):
    """Evaluate the current policy on multiple environments every *N* episodes.

    Also logs Q-value statistics estimated on a sample from the replay buffer
    (capped at ``q_val_max_samples`` to limit overhead).

    Parameters
    ----------
    eval_envs : dict[str, gym.Env]
        Mapping from a logger key (e.g. ``"eval/clean"``) to the gymnasium
        environment used for that evaluation.
    eval_freq : int
        Run evaluation every ``eval_freq`` training episodes.
    n_eval_episodes : int
        Number of episodes to run per evaluation environment.
    deterministic : bool
        Whether to use deterministic actions during evaluation.
    q_val_max_samples : int
        Maximum number of replay buffer samples used for Q-value estimation.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        eval_envs: dict[str, gym.Env],
        eval_freq: int = 10,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        q_val_max_samples: int = 5000,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_envs = eval_envs
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.q_val_max_samples = q_val_max_samples
        self._last_eval_episode = 0
        self._rollout_count = 0

    def _eval_qf_state(self, qf_state, obs, actions):
        """Forward pass through a single critic, returns (n_samples,) Q-values."""
        q_out = qf_state.apply_fn(
            qf_state.params,
            obs,
            actions,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )
        q_out = np.asarray(q_out)
        if q_out.ndim == 3:
            # VectorCritic: (n_critics, n_samples, 1) → min over critics
            return q_out.min(axis=0).squeeze(-1)
        # TQC-style single critic: (n_samples, n_quantiles) → mean over quantiles
        return q_out.mean(axis=-1)

    @property
    def _is_off_policy(self) -> bool:
        return hasattr(self.model, "replay_buffer")

    def _log_q_values(self) -> None:
        if not self._is_off_policy:
            return

        replay_buffer = self.model.replay_buffer
        if replay_buffer is None or replay_buffer.size() == 0:
            return

        n_samples = min(self.q_val_max_samples, replay_buffer.size())
        data = replay_buffer.sample(n_samples)

        obs = data.observations.numpy()
        actions = data.actions.numpy()

        policy = self.model.policy
        if hasattr(policy, "qf_state"):
            # SAC, TD3, CrossQ, DroQ, DDPG: single VectorCritic
            q_values = self._eval_qf_state(policy.qf_state, obs, actions)
        else:
            # TQC: two separate critics, mean matches the policy gradient estimate
            q1 = self._eval_qf_state(policy.qf1_state, obs, actions)
            q2 = self._eval_qf_state(policy.qf2_state, obs, actions)
            q_values = (q1 + q2) / 2

        self.logger.record("eval/q_value_p99", float(np.percentile(q_values, 99)))
        self.logger.record("eval/q_value_mean", float(np.mean(q_values)))

    def _run_eval(self) -> None:
        gamma = self.model.gamma

        for log_key, env in self.eval_envs.items():
            ep_returns, ep_disc_returns, ep_lengths = evaluate_policy(
                self.model,
                env,
                n_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                gamma=gamma,
            )

            self.logger.record(f"{log_key}/mean_return", float(np.mean(ep_returns)))
            self.logger.record(f"{log_key}/std_return", float(np.std(ep_returns)))
            self.logger.record(f"{log_key}/mean_discounted_return", float(np.mean(ep_disc_returns)))
            self.logger.record(f"{log_key}/mean_ep_length", float(np.mean(ep_lengths)))

            if self.verbose >= 1:
                print(
                    f"[Eval step {self.num_timesteps}] {log_key}: "
                    f"mean_return={np.mean(ep_returns):.2f} +/- {np.std(ep_returns):.2f}, "
                    f"mean_discounted_return={np.mean(ep_disc_returns):.2f}"
                )

        self._log_q_values()
        self.logger.dump(step=self.num_timesteps)

    def _on_step(self) -> bool:
        if not self._is_off_policy:
            return True  # On-policy evaluates in _on_rollout_end

        episode_num = self.model._episode_num  # type: ignore[attr-defined]

        if episode_num == self._last_eval_episode:
            return True
        if episode_num % self.eval_freq != 0:
            return True

        self._last_eval_episode = episode_num
        self._run_eval()
        return True

    def _on_rollout_end(self) -> None:
        if self._is_off_policy:
            return  # Off-policy evaluates in _on_step

        self._rollout_count += 1
        if self._rollout_count % self.eval_freq != 0:
            return
        self._run_eval()

    def _on_training_end(self) -> None:
        for env in self.eval_envs.values():
            env.close()
