"""
Smoke tests for Falcon examples.

Run each example for a few epochs to verify the system works end-to-end.
These tests are marked as 'slow' and can be skipped with: pytest -m "not slow"
"""
import pytest
import subprocess
import os
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# Define test cases with their specific configurations
# Each tuple: (example_dir_name, config_name, epoch_overrides)
EXAMPLE_CONFIGS = [
    # 01_minimal: single estimator 'z'
    ("01_minimal", "config.yaml", ["graph.z.estimator.loop.num_epochs=2"]),
    # 02_bimodal: single estimator 'z', using config_regular (needs GPU override)
    ("02_bimodal", "config_regular.yaml", ["graph.z.estimator.loop.num_epochs=2", "graph.z.ray.num_gpus=0"]),
    # 03_composite: two estimators 'z1' and 'z2' (needs GPU override)
    (
        "03_composite",
        "config.yaml",
        [
            "graph.z1.estimator.loop.num_epochs=2",
            "graph.z2.estimator.loop.num_epochs=2",
            "graph.z1.ray.num_gpus=0",
            "graph.z2.ray.num_gpus=0",
        ],
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "example_name,config_name,epoch_overrides",
    EXAMPLE_CONFIGS,
    ids=[f"{e[0]}/{e[1]}" for e in EXAMPLE_CONFIGS],
)
def test_example_runs_without_error(example_name, config_name, epoch_overrides, tmp_path):
    """
    Each example should run for a few epochs without crashing.
    Uses temporary directory for outputs to avoid polluting example dirs.
    """
    example_dir = EXAMPLES_DIR / example_name

    cmd = [
        "falcon",
        "launch",
        f"--config-name={config_name}",
        # Reduce sample counts for faster testing
        "buffer.min_training_samples=64",
        "buffer.max_training_samples=128",
        "buffer.validation_window_size=16",
        "buffer.resample_batch_size=32",
        f"run_dir={tmp_path}",
    ] + epoch_overrides

    # Create a clean environment for the subprocess
    # - Disable WandB logging
    # - Clear RAY_ADDRESS to prevent connecting to existing clusters
    env = {
        **os.environ,
        "WANDB_MODE": "disabled",
        "RAY_ADDRESS": "",  # Force local Ray instance
    }

    result = subprocess.run(
        cmd,
        cwd=example_dir,
        capture_output=True,
        timeout=180,  # 3 minute timeout
        env=env,
    )

    assert result.returncode == 0, (
        f"Example {example_name} with {config_name} failed:\n"
        f"Command: {' '.join(cmd)}\n"
        f"STDOUT:\n{result.stdout.decode()}\n"
        f"STDERR:\n{result.stderr.decode()}"
    )
