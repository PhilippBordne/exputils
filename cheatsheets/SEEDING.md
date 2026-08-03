# Seeding & Reproducibility Cheat Sheet

All patterns used in this codebase to control randomness and ensure deterministic results.

## Compiler / Environment Flags

```python
import os

# Force deterministic GPU operations in JAX/XLA
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + " --xla_gpu_deterministic_ops=true"
    + " --xla_gpu_autotune_level=0"
)

# Disable TensorFloat32 on NVIDIA GPUs (full precision)
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# JAX: use highest precision for matrix multiplications
import jax
jax.config.update("jax_default_matmul_precision", "highest")
```

## Python / NumPy / PyTorch Seeds

```python
import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# CUDA-specific
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# cuDNN determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## JAX PRNG Key Management

```python
import jax

# Create initial key from seed
rng_key = jax.random.PRNGKey(seed)

# Split into multiple independent keys
rng_key, policy_key, alpha_key = jax.random.split(rng_key, 3)

# Consume a key for a single operation, keep the rest
rng_key, consume = jax.random.split(rng_key)
```

## Gymnasium / MuJoCo Environment Seeding

```python
import numpy as np

# Seed the environment on reset after creating, this will set a seed for the environments initial state RNG
# i.e. all subsequent calls to env.reset() will be deterministic
env.reset(seed=seed)

# Seed the action and observation spaces (for sampling)
env.action_space.seed(seed)
env.observation_space.seed(seed)
```

## Checkpointing Checklist

RNG states that need to be saved/restored for exact resumability.

### Global RNGs

- [ ] **Python `random`** — `random.getstate()` / `random.setstate(state)`
- [ ] **NumPy global RNG** — `np.random.get_state()` / `np.random.set_state(state)`
- [ ] **PyTorch CPU RNG** — `torch.random.get_rng_state()` / `torch.random.set_rng_state(state)`
- [ ] **PyTorch CUDA RNG** — `torch.cuda.get_rng_state_all()` / `torch.cuda.set_rng_state_all(states)`

### Object-level RNGs

- [ ] **JAX PRNG key** — the key you keep splitting from; save/restore the array directly
- [ ] **NumPy `default_rng` instances** — `rng.bit_generator.state` (e.g. noise RNGs in env wrappers). Save/restore:
  ```python
  state = rng.bit_generator.state      # save
  rng.bit_generator.state = state      # restore
  ```
- [ ] **Gymnasium env** — `env.np_random.bit_generator.state` (a `np.random.Generator`)
- [ ] **Gymnasium spaces** — `env.action_space.np_random.bit_generator.state` and `env.observation_space.np_random.bit_generator.state` (each space has its own `np.random.Generator`, set via `space.seed()`)
