# Logging Setup for SB3 / SBX

Route SB3/SBX logging to **W&B + CSV + STDOUT**, no TensorBoard.

## Components

| Component | Role |
|---|---|
| `Logger` | Single hub. Holds a pending key-value dict and a list of output formats. |
| `record(k, v)` | Stage a value. Overwrites within a dump window (last wins). |
| `record_mean(k, v)` | Stage a value; emits the mean over the window at dump. |
| `dump(step)` | Fan out pending values to *every* output format, then clear. |
| output format (`KVWriter`) | A destination. `CSVOutputFormat`, `HumanOutputFormat`, custom `WandbWriter` — all peers. |
| `BaseCallback` | User hook. `self.logger` is the same `Logger`. |

**Relation:** `record` → one `Logger` → `dump` fans out to N output formats.
W&B is just an output format (no `"wandb"` string tag exists, so it's added as an explicit instance).

## Dump cadence

SB3 calls `dump` itself, gated by `learn(log_interval=...)`:
- On-policy (PPO/A2C): per rollout, every `log_interval` rollouts (default 1).
- Off-policy (SAC/TD3/DQN): every `log_interval` episodes (default 4).

## Setup

```python
import sys, wandb
from stable_baselines3 import PPO
from stable_baselines3.common.logger import (
    KVWriter, Logger, CSVOutputFormat, HumanOutputFormat,
)


class WandbWriter(KVWriter):
    def write(self, key_values, key_excluded, step=0):
        wandb.log({k: v for k, v in key_values.items()
                   if isinstance(v, (int, float))}, step=step)  # scalars only

    def close(self):
        wandb.finish()


wandb.init(project="my-proj", config={...})
model = PPO("MlpPolicy", env)
model.set_logger(Logger(
    folder="./logs",                       # CSV needs a real folder
    output_formats=[
        CSVOutputFormat("./logs/progress.csv"),
        WandbWriter(),
        HumanOutputFormat(sys.stdout),     # drop for no console output
    ],
))
model.learn(total_timesteps=1_000_000, log_interval=1)
```

## Custom metrics (callback)

```python
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class MyStatsCallback(BaseCallback):
    def _on_training_start(self):
        self._buf = deque(maxlen=100)

    def _on_step(self) -> bool:
        v = ...
        self.logger.record_mean("train/my_stat", v)              # mean over window
        self._buf.append(v)
        self.logger.record("train/my_stat_win", float(np.mean(self._buf)))  # fixed window
        return True  # False stops training


model.learn(total_timesteps=N, callback=[MyStatsCallback()])  # list -> auto CallbackList
```

Hooks: `_on_training_start` · `_on_rollout_start` · `_on_step` (False=stop) ·
`_on_rollout_end` · `_on_training_end`. No eval hook — `EvalCallback` runs eval
in `_on_step` gated by `eval_freq`.

## Default reward metric

`ep_rew_mean`/`ep_len_mean`: mean over a 100-episode buffer, fed by the
`Monitor` wrapper's `info["episode"]`. Add a new metric under your own key;
redefine the canonical one by intercepting in a custom `Monitor`.

## Misc: rollout videos to W&B

`monitor_gym=True` needs the TB bridge (and is reported broken in SB3 ≥ 2.4.0).
TB-free: render frames and log directly.

```python
class VideoEvalCallback(BaseCallback):
    def __init__(self, eval_env, freq=50_000, length=500):  # env: render_mode="rgb_array"
        super().__init__()
        self.eval_env, self.freq, self.length = eval_env, freq, length

    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0:
            frames, (obs, _) = [], self.eval_env.reset()
            for _ in range(self.length):
                a, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = self.eval_env.step(a)
                frames.append(self.eval_env.render())
                if term or trunc:
                    break
            video = np.stack(frames).transpose(0, 3, 1, 2)  # (T,H,W,C)->(T,C,H,W)
            wandb.log({"rollout/video": wandb.Video(video, fps=30)}, step=self.num_timesteps)
        return True
```

Or write mp4 with `VecVideoRecorder` (needs `ffmpeg`) and upload the file.