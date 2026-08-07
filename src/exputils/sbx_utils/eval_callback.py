"""Episode-based evaluation callback for SBX / SB3 off-policy algorithms.

Evaluates the agent on one or more environments at a fixed episode interval,
matching the ``log_interval`` semantics of off-policy algorithms (which count
episodes, not timesteps).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class EvalCallback(BaseCallback):
    """Evaluate the current policy on multiple environments every *N* episodes.

    Parameters
    ----------
    eval_envs : dict[str, gym.Env]
        Mapping from a logger key (e.g. ``"eval/return_noisy"``) to the
        gymnasium environment used for that evaluation.
    eval_freq : int
        Run evaluation every ``eval_freq`` training episodes.  Uses the same
        episode counter as the off-policy ``log_interval``.
    n_eval_episodes : int
        Number of episodes to run per evaluation environment.
    deterministic : bool
        Whether to use deterministic actions during evaluation.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        eval_envs: dict[str, gym.Env],
        eval_freq: int = 10,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_envs = eval_envs
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self._last_eval_episode = 0

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

        self.logger.dump(step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        for env in self.eval_envs.values():
            env.close()
