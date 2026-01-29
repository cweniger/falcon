#!/usr/bin/env python3
"""Run the linear regression example and compare with analytic posterior.

This script:
1. Generates mock observation data (if needed)
2. Runs falcon launch to train the model
3. Generates posterior samples
4. Compares inferred posterior with analytic solution

Model: y = Phi @ theta + noise
  - Phi[i, k] = sin((k+1) * x_i), 100 bins, 10 parameters
  - Prior: theta ~ N(0, I)
  - Noise: N(0, 0.1^2 * I)
"""

import subprocess
import sys
from pathlib import Path
import numpy as np

# Configuration
RUN_DIR = "outputs/run"


def run_command(cmd: list[str], description: str, cwd: Path = None) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    return True


def generate_mock_data(script_dir: Path) -> bool:
    """Generate mock data if it doesn't exist."""
    mock_data_path = script_dir / "data" / "mock_data.npz"
    if mock_data_path.exists():
        print(f"Mock data already exists: {mock_data_path}")
        return True

    return run_command(
        ["python", "gen_mock_data.py"],
        "Generating mock observation data",
        cwd=script_dir / "data"
    )


def analyze_samples(samples_dir: Path, script_dir: Path) -> None:
    """Load posterior samples and compare with analytic posterior."""
    print(f"\n{'='*60}")
    print("Comparing inferred vs analytic posterior")
    print(f"{'='*60}")

    # Load analytic posterior
    mock_data = np.load(script_dir / "data" / "mock_data.npz")
    theta_true = mock_data["theta_true"]
    mu_post_analytic = mock_data["mu_post"]
    Sigma_post_analytic = mock_data["Sigma_post"]
    marginal_std_analytic = mock_data["marginal_std"]

    # Find posterior samples
    posterior_dir = samples_dir / "posterior"
    if not posterior_dir.exists():
        print(f"No posterior samples found in {posterior_dir}")
        return

    timestamp_dirs = sorted(posterior_dir.iterdir())
    if not timestamp_dirs:
        print("No sample batches found")
        return

    latest_dir = timestamp_dirs[-1]
    print(f"Loading samples from: {latest_dir}")

    # Load all sample files
    all_samples = []
    for npz_file in sorted(latest_dir.glob("*.npz")):
        data = np.load(npz_file)
        if 'theta' in data:
            all_samples.append(data['theta'])

    if not all_samples:
        print("No 'theta' samples found in npz files")
        return

    samples = np.concatenate(all_samples, axis=0) if len(all_samples) > 1 else all_samples[0]
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n_samples, n_params = samples.shape
    print(f"Loaded {n_samples} samples with {n_params} parameters\n")

    # Inferred statistics
    means_inferred = np.mean(samples, axis=0)
    stds_inferred = np.std(samples, axis=0)

    # Print comparison table
    print(f"{'Param':<10} {'True':>8} {'Analytic':>10} {'Inferred':>10} {'Anal Std':>10} {'Inf Std':>10}")
    print("-" * 60)
    for k in range(n_params):
        print(f"theta[{k}]  {theta_true[k]:>8.4f} {mu_post_analytic[k]:>10.4f} "
              f"{means_inferred[k]:>10.4f} {marginal_std_analytic[k]:>10.6f} "
              f"{stds_inferred[k]:>10.6f}")

    # Summary
    mean_error = np.sqrt(np.mean((means_inferred - mu_post_analytic)**2))
    std_ratio = stds_inferred / marginal_std_analytic
    print(f"\nRMSE(mean_inferred - mean_analytic): {mean_error:.6f}")
    print(f"Std ratio (inferred/analytic): {std_ratio.mean():.4f} +/- {std_ratio.std():.4f}")


def main():
    script_dir = Path(__file__).parent
    run_dir = script_dir / RUN_DIR

    # Step 0: Generate mock data if needed
    if not generate_mock_data(script_dir):
        sys.exit(1)

    # Step 1: Run falcon launch
    if not run_command(
        ["falcon", "launch", f"--run-dir={RUN_DIR}"],
        "Step 1: Running falcon launch",
        cwd=script_dir
    ):
        sys.exit(1)

    # Step 2: Generate posterior samples
    if not run_command(
        ["falcon", "sample", "posterior", f"--run-dir={RUN_DIR}"],
        "Step 2: Generating posterior samples",
        cwd=script_dir
    ):
        sys.exit(1)

    # Step 3: Analyze samples
    samples_dir = run_dir / "samples_dir"
    analyze_samples(samples_dir, script_dir)


if __name__ == "__main__":
    main()
