import numpy as np

SIGMA = 1e-3
NBINS = 100000

# (1, NBINS) with SIGMA std, derived from standard normal distribution * SIGMA
x = np.random.randn(1, NBINS) * SIGMA

np.savez('obs_100000.npz', x=x)
