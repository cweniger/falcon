#!/usr/bin/env python
"""
Usage: python make_plots.py RUN_PATH
"""
import sys
import corner
import falcon

run = falcon.load_run(sys.argv[1])
z = run.samples.posterior.stacked['z']
obs = run.observations.get('x')

corner.corner(z, truths=obs, show_titles=True)
corner.pyplot.show()
