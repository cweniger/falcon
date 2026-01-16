#!/usr/bin/env python
"""
Usage: python make_plots.py RUN_PATH [SAMPLE_TYPE]

SAMPLE_TYPE: posterior (default), prior, or proposal

For posterior plots, loads ground truth from mock_data.npz['z_truth'] if available.
"""
import sys
import numpy as np
import corner
import falcon

run = falcon.load_run(sys.argv[1])
sample_type = sys.argv[2] if len(sys.argv) > 2 else "posterior"

samples = getattr(run.samples, sample_type)
z = samples.stacked['z']

# Load ground truth for posterior validation
truths = None
if sample_type == "posterior":
    # Try to load z_truth from the mock data NPZ file
    mock_data_path = run.run_dir.parent / "data/mock_data.npz"
    if mock_data_path.exists():
        mock_data = np.load(mock_data_path)
        if 'z_truth' in mock_data.files:
            truths = mock_data['z_truth']

fig = corner.corner(z, truths=truths, show_titles=True)
fig.savefig(run.run_dir / f"corner_{sample_type}.png", dpi=150)
print(f"Saved to {run.run_dir / f'corner_{sample_type}.png'}")
