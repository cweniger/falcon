"""
Smoke tests for Falcon examples.

Run each example for a few epochs to verify the system works end-to-end.
These tests are marked as 'slow' and can be skipped with: pytest -m "not slow"
"""
import pytest
import subprocess
import os
import numpy as np
import joblib
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# Define test cases with their specific configurations
# Each tuple: (example_dir_name, config_name, epoch_overrides)
EXAMPLE_CONFIGS = [
    # 01_minimal: single estimator 'z'
    ("01_minimal", "config", ["graph.z.estimator.num_epochs=2"]),
    # 02_bimodal: single estimator 'z', using config_regular (needs GPU override)
    ("02_bimodal", "config_regular", ["graph.z.estimator.num_epochs=2", "graph.z.ray.num_gpus=0"]),
    # 03_composite: two estimators 'z1' and 'z2' (needs GPU override)
    (
        "03_composite",
        "config",
        [
            "graph.z1.estimator.num_epochs=2",
            "graph.z2.estimator.num_epochs=2",
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
        f"hydra.run.dir={tmp_path}",
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


@pytest.mark.slow
@pytest.mark.gpu
def test_01_minimal_posterior_quality(tmp_path):
    """
    Validation test for 01_minimal: verify posterior inference quality.

    The model is: x = z + N(0, sigma) with sigma=0.1 and uniform prior U(-100,100).
    The prior std is ~57.7 (uniform over 200 range).

    This test verifies that the learned posterior is significantly narrower than
    the prior, indicating the network has learned something meaningful.

    This test:
    1. Trains the model for enough epochs to show learning
    2. Samples from the posterior
    3. Verifies that the posterior std is much smaller than prior std
    """
    example_dir = EXAMPLES_DIR / "01_minimal"

    # Prior is U(-100, 100), so prior_std = 200/sqrt(12) ≈ 57.7
    # Expected posterior std is ~0.1, but with limited training we may not converge
    # We check that posterior is at least 10x narrower than prior to catch failures
    prior_std = 200 / np.sqrt(12)  # ~57.7
    max_acceptable_std = prior_std / 10  # ~5.77 - should be much narrower than prior

    env = {
        **os.environ,
        "WANDB_MODE": "disabled",
        "RAY_ADDRESS": "",
    }

    # Step 1: Train the model
    # Balance between convergence quality and test runtime
    train_cmd = [
        "falcon",
        "launch",
        "--config-name=config",
        "buffer.min_training_samples=1024",
        "buffer.max_training_samples=4096",
        "buffer.validation_window_size=128",
        "buffer.resample_batch_size=64",
        "graph.z.estimator.num_epochs=150",
        "graph.z.ray.num_gpus=1",
        f"hydra.run.dir={tmp_path}",
    ]

    train_result = subprocess.run(
        train_cmd,
        cwd=example_dir,
        capture_output=True,
        timeout=420,  # 7 minute timeout for training
        env=env,
    )

    assert train_result.returncode == 0, (
        f"Training failed:\n"
        f"Command: {' '.join(train_cmd)}\n"
        f"STDOUT:\n{train_result.stdout.decode()}\n"
        f"STDERR:\n{train_result.stderr.decode()}"
    )

    # Step 2: Sample from posterior
    # Must use GPU since model was trained on GPU
    samples_path = tmp_path / "samples_posterior.joblib"
    sample_cmd = [
        "falcon",
        "sample",
        "posterior",
        "--config-name=config",
        "sample.posterior.n=1000",
        "graph.z.ray.num_gpus=1",
        f"sample.posterior.path={samples_path}",
        f"hydra.run.dir={tmp_path}",
    ]

    sample_result = subprocess.run(
        sample_cmd,
        cwd=example_dir,
        capture_output=True,
        timeout=120,  # 2 minute timeout for sampling
        env=env,
    )

    assert sample_result.returncode == 0, (
        f"Sampling failed:\n"
        f"Command: {' '.join(sample_cmd)}\n"
        f"STDOUT:\n{sample_result.stdout.decode()}\n"
        f"STDERR:\n{sample_result.stderr.decode()}"
    )

    # Step 3: Load and analyze posterior samples
    assert samples_path.exists(), f"Samples file not found at {samples_path}"

    samples = joblib.load(samples_path)
    # samples is a list of dicts, each with 'z' key containing array of shape (n_params,)
    assert isinstance(samples, list), f"Expected list, got {type(samples)}"
    assert len(samples) > 0, "Empty samples list"
    assert "z" in samples[0], f"Expected 'z' in samples, got keys: {samples[0].keys()}"

    # Stack samples into a 2D array (n_samples, n_params)
    z_samples = np.stack([s["z"] for s in samples])

    # Compute empirical std for each parameter
    empirical_std = np.std(z_samples, axis=0)
    mean_empirical_std = np.mean(empirical_std)

    # Print diagnostic info (visible with pytest -v)
    print(f"\nPosterior std per parameter: {empirical_std}")
    print(f"Mean posterior std: {mean_empirical_std:.4f} (expected ~0.1, max acceptable: {max_acceptable_std:.2f})")

    # Verify posterior is significantly narrower than the prior
    # This catches gross failures while allowing for imperfect convergence
    for i, std_i in enumerate(empirical_std):
        assert std_i <= max_acceptable_std, (
            f"Parameter z[{i}] posterior std {std_i:.4f} too wide "
            f"(max acceptable: {max_acceptable_std:.4f}, prior std: {prior_std:.4f})"
        )
