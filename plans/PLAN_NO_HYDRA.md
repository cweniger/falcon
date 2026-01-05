# Plan: Remove Hydra, Replace with OmegaConf + jsonargparse

## Current Hydra Usage

### 1. Entry Points (`falcon/cli.py`)
```python
@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def launch_main(cfg: DictConfig) -> None:
    launch_mode(cfg)
```

### 2. Hydra-specific Interpolation (`${hydra:run.dir}`)
Used in all example configs for output paths:
```yaml
paths:
  buffer: ${hydra:run.dir}/sim_dir
  graph: ${hydra:run.dir}/graph_dir
logging:
  wandb:
    dir: ${hydra:run.dir}
```

### 3. CLI Override Syntax
```bash
falcon launch buffer.num_epochs=500 hydra.run.dir=outputs/exp01
```

### 4. Config Selection
```bash
falcon launch --config-name config_amortized
```

---

## Migration Plan

### Step 1: Replace `${hydra:run.dir}` with `${run_dir}`

**Config changes** (all example YAML files):
```yaml
# Before
paths:
  buffer: ${hydra:run.dir}/sim_dir

# After
run_dir: ./outputs/${now:%Y-%m-%d_%H-%M-%S}  # Default, can be overridden

paths:
  buffer: ${run_dir}/sim_dir
```

Register custom OmegaConf resolver for `now`:
```python
from datetime import datetime
OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))
```

### Step 2: Replace `@hydra.main()` with Manual Loading

**New `falcon/cli.py`** structure:

```python
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from datetime import datetime

# Register custom resolvers
OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))

def load_config(config_name: str = "config", overrides: list = None) -> DictConfig:
    """Load config from current directory with CLI overrides."""
    config_path = Path.cwd() / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Apply CLI overrides (key=value format)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Resolve all interpolations
    OmegaConf.resolve(cfg)

    # Create run_dir if it doesn't exist
    run_dir = cfg.get("run_dir")
    if run_dir:
        Path(run_dir).mkdir(parents=True, exist_ok=True)

    return cfg


def parse_args():
    """Parse falcon CLI arguments."""
    # falcon launch [--config-name=X] [overrides...]
    # falcon sample prior/posterior/proposal [--config-name=X] [overrides...]

    if len(sys.argv) < 2 or sys.argv[1] not in ["sample", "launch"]:
        print("Usage:")
        print("  falcon launch [--config-name=X] [key=value ...]")
        print("  falcon sample prior|posterior|proposal [--config-name=X] [key=value ...]")
        sys.exit(1)

    mode = sys.argv[1]
    args = sys.argv[2:]

    sample_type = None
    if mode == "sample":
        if not args or args[0] not in ["prior", "posterior", "proposal"]:
            print("Error: sample requires type: prior, posterior, or proposal")
            sys.exit(1)
        sample_type = args.pop(0)

    # Extract --config-name
    config_name = "config"
    remaining = []
    for arg in args:
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
        elif arg.startswith("--config-name"):
            # Handle --config-name X format (next iteration)
            pass  # Simplified: only support = format
        else:
            remaining.append(arg)

    return mode, sample_type, config_name, remaining


def main():
    mode, sample_type, config_name, overrides = parse_args()
    cfg = load_config(config_name, overrides)

    if mode == "launch":
        launch_mode(cfg)
    else:
        sample_mode(cfg, sample_type)
```

### Step 3: Update Example Configs

Replace all `${hydra:run.dir}` with `${run_dir}`:

```yaml
# Add at top of each config
run_dir: ./outputs/${now:%Y-%m-%d_%H-%M-%S}

# Replace all occurrences
paths:
  buffer: ${run_dir}/sim_dir      # was ${hydra:run.dir}/sim_dir
  graph: ${run_dir}/graph_dir     # was ${hydra:run.dir}/graph_dir

logging:
  wandb:
    dir: ${run_dir}               # was ${hydra:run.dir}
```

### Step 4: Update Tests

Change `hydra.run.dir=` to `run_dir=`:

```python
# tests/test_examples_smoke.py
cmd = [
    "falcon", "launch",
    f"--config-name={config_name}",
    f"run_dir={tmp_path}",  # was hydra.run.dir=
    ...
]
```

### Step 5: Update Dependencies

**setup.py**:
```python
install_requires=[
    # Remove: "hydra-core",
    "omegaconf",      # Keep (already there)
    # Add if needed for advanced CLI: "jsonargparse",
    ...
]
```

---

## Optional: Add jsonargparse for Type-Safe CLI

If you want automatic CLI generation from config schema:

```python
from jsonargparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", type=Path, default="config.yaml")
parser.add_argument("--run_dir", type=Path)
parser.add_argument("--buffer.min_training_samples", type=int)
# ... etc

args = parser.parse_args()
```

For now, simple `OmegaConf.from_dotlist()` is sufficient and maintains the same CLI syntax.

---

## Files to Modify

| File | Change |
|------|--------|
| `falcon/cli.py` | Replace `@hydra.main()` with manual loading |
| `setup.py` | Remove `hydra-core` dependency |
| `examples/01_minimal/config.yaml` | Replace `${hydra:run.dir}` → `${run_dir}` |
| `examples/02_bimodal/config_*.yaml` | Replace `${hydra:run.dir}` → `${run_dir}` |
| `examples/03_composite/config.yaml` | Replace `${hydra:run.dir}` → `${run_dir}` |
| `tests/test_examples_smoke.py` | Replace `hydra.run.dir=` → `run_dir=` |
| `falcon/core/logger.py` | Update docstring examples |

---

## CLI Compatibility

| Before (Hydra) | After (OmegaConf) |
|----------------|-------------------|
| `falcon launch` | `falcon launch` (same) |
| `falcon launch --config-name=X` | `falcon launch --config-name=X` (same) |
| `falcon launch key=value` | `falcon launch key=value` (same) |
| `falcon launch hydra.run.dir=X` | `falcon launch run_dir=X` |
| `falcon sample posterior` | `falcon sample posterior` (same) |

Only breaking change: `hydra.run.dir=` → `run_dir=`

---

## Execution Order

1. Update `falcon/cli.py` (core change)
2. Update all example configs (simple find/replace)
3. Update tests
4. Update `setup.py` dependencies
5. Test all examples
6. Update documentation/comments

Estimated lines of code changed: ~100
