"""
Smoke tests for Falcon examples.

Run each example for a couple of short rounds to verify the system works end-to-end.
These tests are marked as 'slow' and can be skipped with: pytest -m "not slow"
"""
import pytest
import subprocess
import os
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
IN_CI = os.environ.get("CI") == "true"

_skip_ci = pytest.mark.skipif(IN_CI, reason="Too resource-heavy for CI runners")

# Prefix of a config name: run the config with ray.num_gpus instead of the split GPU keys
LEGACY_GPUS = "legacy_gpus:"

# Define test cases with their specific configurations
# Each tuple: (example_dir_name, config_name, epoch_overrides)
EXAMPLE_CONFIGS = [
    # 01_minimal: single estimator 'z'
    ("01_minimal", "config.yml", ["graph.z.estimator.max_epochs=2", "graph.z.estimator.max_rounds=2"]),
    # 01_minimal with the deprecated ray.num_gpus, as in configs saved before the actor split
    pytest.param(
        "01_minimal", LEGACY_GPUS + "config.yml",
        ["graph.z.estimator.max_epochs=2", "graph.z.estimator.max_rounds=2"],
        id="01_minimal/legacy_num_gpus",
    ),
    # 01_minimal with FlowMatching: small networks and region draws for CPU runners
    ("01_minimal", "config_flow_matching.yml",
     ["graph.z.estimator.max_epochs=2", "graph.z.estimator.max_rounds=2",
      "graph.z.estimator.val_every_epochs=1", "graph.z.estimator.hidden=32", "graph.z.estimator.layers=2",
      "graph.z.estimator.n_region=1024", "graph.z.estimator.n_mout=1024",
      "graph.z.estimator.v_max_draws=4096", "graph.z.estimator.readout_draws=1024",
      "graph.z.ray.num_train_gpus=0", "graph.z.ray.num_sample_gpus=0"]),
    # 02_bimodal: single estimator 'z', using config_regular (needs GPU override)
    ("02_bimodal", "config_regular.yml", ["graph.z.estimator.max_epochs=2", "graph.z.estimator.max_rounds=2",
                                          "graph.z.ray.num_train_gpus=0", "graph.z.ray.num_sample_gpus=0"]),
    # 03_composite: two ResNet18 + Ray actors exceed CI runner memory
    pytest.param(
        "03_composite", "config.yml",
        ["graph.z1.estimator.max_epochs=2", "graph.z1.estimator.max_rounds=2",
         "graph.z2.estimator.max_epochs=2", "graph.z2.estimator.max_rounds=2",
         "graph.z1.ray.num_train_gpus=0", "graph.z1.ray.num_sample_gpus=0",
         "graph.z2.ray.num_train_gpus=0", "graph.z2.ray.num_sample_gpus=0"],
        marks=_skip_ci,
    ),
    # 04_gaussian: SNPE_gaussian with exponential forward model (needs GPU override)
    ("04_gaussian", "config.yml", ["graph.z.estimator.max_epochs=2", "graph.z.estimator.max_rounds=2",
                                   "graph.z.ray.num_train_gpus=0", "graph.z.ray.num_sample_gpus=0"]),
    # 05_linear_regression: requires GPU
    pytest.param(
        "05_linear_regression", "config.yml",
        ["graph.theta.estimator.max_epochs=2", "graph.theta.estimator.max_rounds=2",
         "graph.theta.ray.num_train_gpus=0", "graph.theta.ray.num_sample_gpus=0"],
        marks=_skip_ci,
    ),
]


def _case_id(case):
    if hasattr(case, "values"):  # pytest.param
        return case.id or f"{case.values[0]}/{case.values[1]}"
    return f"{case[0]}/{case[1]}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "example_name,config_name,epoch_overrides",
    EXAMPLE_CONFIGS,
    ids=[_case_id(case) for case in EXAMPLE_CONFIGS],
)
def test_example_runs_without_error(example_name, config_name, epoch_overrides, tmp_path):
    """
    Each example should run for a few epochs without crashing.
    Uses temporary directory for outputs to avoid polluting example dirs.
    """
    example_dir = EXAMPLES_DIR / example_name
    if config_name.startswith(LEGACY_GPUS):
        text = (example_dir / config_name[len(LEGACY_GPUS):]).read_text()
        text = text.replace("num_train_gpus: 0 ", "num_gpus: 0 ")
        text = "\n".join(line for line in text.splitlines() if "num_sample_gpus" not in line)
        assert "num_gpus: 0" in text
        legacy_config = tmp_path.parent / f"{tmp_path.name}_config.yml"
        legacy_config.write_text(text)
        config_name = str(legacy_config)  # absolute, so the example dir stays the cwd

    cmd = [
        "falcon",
        "launch",
        f"--config={config_name}",
        f"--output={tmp_path}",
        # Reduce sample counts for faster testing
        "buffer.min_samples=64",
        "buffer.max_samples=128",
        "buffer.simulate_count=32",
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
    graph_dir = tmp_path / "graph"
    assert graph_dir.exists(), f"graph dir not found at {graph_dir}"

    # Check driver output.log exists
    driver_log = graph_dir / "driver" / "output.log"
    assert driver_log.exists(), f"Driver output.log not found at {driver_log}"

    # Every estimator node has a sample actor and a train actor, each with its own log,
    # and the train actor wrote the checkpoint
    estimator_nodes = [o.split(".")[1] for o in epoch_overrides if o.endswith(".estimator.max_rounds=2")]
    for node in estimator_nodes:
        for log in (graph_dir / node / "output.log", graph_dir / node / "train" / "output.log"):
            assert log.exists(), (
                f"{log} not found. Found: {sorted(str(p.relative_to(graph_dir)) for p in graph_dir.rglob('output.log'))}"
            )
        assert (graph_dir / node / "best_state.npz").exists()

    # Check driver/output.log exists with runtime logging
    driver_log = graph_dir / "driver" / "output.log"
    assert driver_log.exists(), f"driver/output.log not found at {driver_log}"
    driver_log_content = driver_log.read_text()
    assert len(driver_log_content) > 0, "driver/output.log is empty"
    # Verify it contains timestamped log entries
    assert "[INFO]" in driver_log_content, "driver/output.log missing INFO level entries"
