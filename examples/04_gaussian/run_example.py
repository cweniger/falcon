#!/usr/bin/env python3
"""Run the Gaussian posterior example and analyze results.

This script:
1. Generates mock observation data (if needed)
2. Runs falcon launch to train the model
3. Generates posterior samples
4. Calculates and prints mean/std for each parameter

Ground truth: z = [-5, 0, 5]
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


def analyze_samples(samples_dir: Path) -> None:
    """Load and analyze posterior samples."""
    print(f"\n{'='*60}")
    print("Analyzing posterior samples")
    print(f"{'='*60}")

    # Find the most recent posterior samples
    posterior_dir = samples_dir / "posterior"
    if not posterior_dir.exists():
        print(f"No posterior samples found in {posterior_dir}")
        return

    # Get the most recent timestamp directory
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
        print(f"  {npz_file.name}: keys={list(data.keys())}")
        # The samples are stored under the node name 'z'
        if 'z' in data:
            arr = data['z']
            print(f"    z: shape={arr.shape}")
            all_samples.append(arr)

    if not all_samples:
        print("No samples found in npz files")
        return

    samples = np.concatenate(all_samples, axis=0) if len(all_samples) > 1 else all_samples[0]

    # Ensure 2D shape
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n_samples, n_params = samples.shape
    print(f"Loaded {n_samples} samples with {n_params} parameters")

    # Calculate statistics
    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)

    # Ground truth
    z_true = np.array([-5.0, 0.0, 5.0])

    print(f"\n{'Parameter':<12} {'Mean':>10} {'Std':>10} {'True':>10} {'Error':>10}")
    print("-" * 54)
    for i, (m, s, t) in enumerate(zip(means, stds, z_true)):
        error = m - t
        print(f"z[{i}]         {m:>10.4f} {s:>10.4f} {t:>10.4f} {error:>+10.4f}")

    print(f"\nRMSE: {np.sqrt(np.mean((means - z_true)**2)):.4f}")


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
    analyze_samples(samples_dir)


if __name__ == "__main__":
    main()
