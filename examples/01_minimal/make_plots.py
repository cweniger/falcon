#!/usr/bin/env python
"""
Usage: python make_plots.py RUN_PATH [SAMPLE_TYPE]

SAMPLE_TYPE: posterior (default), prior, or proposal
"""
import sys
import corner
import falcon

run = falcon.load_run(sys.argv[1])
sample_type = sys.argv[2] if len(sys.argv) > 2 else "posterior"

samples = getattr(run.samples, sample_type)
z = samples.stacked['z']
obs = run.observations.get('x') if sample_type == "posterior" else None

fig = corner.corner(z, truths=obs, show_titles=True)
fig.savefig(run.run_dir / f"corner_{sample_type}.png", dpi=150)
print(f"Saved to {run.run_dir / f'corner_{sample_type}.png'}")
