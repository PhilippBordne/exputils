"""Episode-based evaluation callback for SBX / SB3 off-policy algorithms.

Evaluates the agent on one or more environments at a fixed episode interval,
matching the ``log_interval`` semantics of off-policy algorithms (which count
episodes, not timesteps).
"""

from __future__ import annotations

import gymnasium as gym
import jax
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class EvalCallback(BaseCallback):
    """Evaluate the current policy on multiple environments every *N* episodes.

    Also logs the 90th-percentile of Q-values estimated on a sample from the
    replay buffer (capped at ``q_val_max_samples`` to limit overhead).

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

    def _eval_qf_state(self, qf_state, obs, actions):
        """Forward pass through a single critic, returns (n_samples,) Q-values."""
        q_out = qf_state.apply_fn(
            qf_state.params, obs, actions,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )
        q_out = np.asarray(q_out)
        if q_out.ndim == 3:
            # VectorCritic: (n_critics, n_samples, 1) → min over critics
            return q_out.min(axis=0).squeeze(-1)
        # TQC-style single critic: (n_samples, n_quantiles) → mean over quantiles
        return q_out.mean(axis=-1)

    def _log_q_values(self) -> None:
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

        self.logger.record("eval/q_value_p90", float(np.percentile(q_values, 90)))
        self.logger.record("eval/q_value_mean", float(np.mean(q_values)))

    def _on_step(self) -> bool:
        episode_num = self.model._episode_num  # type: ignore[attr-defined]

        if episode_num == self._last_eval_episode:
            return True
        if episode_num % self.eval_freq != 0:
            return True

        self._last_eval_episode = episode_num

        for log_key, env in self.eval_envs.items():
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=False,
            )
            mean_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))
            mean_length = float(np.mean(episode_lengths))

            self.logger.record(f"{log_key}/mean_return", mean_reward)
            self.logger.record(f"{log_key}/std_return", std_reward)
            self.logger.record(f"{log_key}/mean_ep_length", mean_length)

            if self.verbose >= 1:
                print(
                    f"[Eval episode {episode_num}] {log_key}: "
                    f"mean_return={mean_reward:.2f} +/- {std_reward:.2f}, "
                    f"mean_ep_length={mean_length:.0f}"
                )

        self._log_q_values()
        self.logger.dump(step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        for env in self.eval_envs.values():
            env.close()
