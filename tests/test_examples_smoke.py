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
    # 04_gaussian: SNPE_gaussian with exponential forward model (needs GPU override)
    ("04_gaussian", "config.yaml", ["graph.z.estimator.loop.num_epochs=2", "graph.z.ray.num_gpus=0"]),
    # 05_linear_regression: SNPE_gaussian with linear regression (needs GPU override)
    ("05_linear_regression", "config.yaml", ["graph.theta.estimator.loop.num_epochs=2", "graph.theta.ray.num_gpus=0"]),
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
        f"--run-dir={tmp_path}",
        # Reduce sample counts for faster testing
        "buffer.min_training_samples=64",
        "buffer.max_training_samples=128",
        "buffer.validation_window_size=16",
        "buffer.resample_batch_size=32",
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

    # Verify output.log files were created by the logging system
    graph_dir = tmp_path / "graph_dir"
    assert graph_dir.exists(), f"graph_dir not found at {graph_dir}"

    # Check driver output.log exists
    driver_log = graph_dir / "driver" / "output.log"
    assert driver_log.exists(), f"Driver output.log not found at {driver_log}"

    # Check that at least one node has output.log (actor logging works)
    node_logs = list(graph_dir.glob("*/output.log"))
    # Filter out driver to check actor logs specifically
    actor_logs = [p for p in node_logs if p.parent.name != "driver"]
    assert len(actor_logs) > 0, (
        f"No actor output.log files found in {graph_dir}. "
        f"Found directories: {[p.name for p in graph_dir.iterdir() if p.is_dir()]}"
    )

    # Check driver/output.log exists with runtime logging
    driver_log = graph_dir / "driver" / "output.log"
    assert driver_log.exists(), f"driver/output.log not found at {driver_log}"
    driver_log_content = driver_log.read_text()
    assert len(driver_log_content) > 0, "driver/output.log is empty"
    # Verify it contains timestamped log entries
    assert "[INFO]" in driver_log_content, "driver/output.log missing INFO level entries"
