## pyproject.toml
```toml
[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "sbx-rl",
    "exputils",
    "numpy",
    "jax",
]

# specify dependencies to be resolved from local paths/submodules
[tool.uv.sources]
sbx-rl      = { path = "externals/sbx",      editable = true }
exputils = { path = "externals/exputils", editable = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
# everything in src/ is included in wheel
only-include = ["src"]
sources = {"src" = ""}
# specific package(s)
# packages = ["src/myproject", ...]
```

## uv commands
```bash
### Environment setup
uv sync                  # create/update .venv to match uv.lock (re-resolves if lock is stale)
uv sync --locked         # same, but error if uv.lock is out of date (does not match pyproject toml) — use for reproducibility/CI
uv sync --upgrade        # re-resolve to newest allowed versions, then sync
uv venv                  # create .venv only, without installing

### Managing dependencies
# only ways to modify the pyproject.toml through uv; 
uv add <pkg>             # add dependency to pyproject.toml + update lock + sync venv
uv add "<pkg>==1.2.3"    # add with an explicit version constraint
uv add -r requirements.txt   # bulk-add from a requirements file
uv remove <pkg>          # remove dependency from all three

### Lockfile
# will install nothing
uv lock                  # re-resolve pyproject.toml → uv.lock (deterministic; prefers existing lock)
uv lock --upgrade        # re-resolve to newest allowed versions
uv lock --upgrade-package <pkg>   # upgrade a single package

### Running
uv run <cmd>             # sync (lock + venv), then run cmd inside the env
uv run python script.py  # e.g. run a script

### Inspection
uv pip freeze            # list installed packages in the venv
uv tree                  # show the dependency tree
```
