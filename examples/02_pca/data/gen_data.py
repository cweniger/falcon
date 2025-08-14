import numpy as np

SIGMA = 1e-3
NBINS = 10000

# (1, NBINS) with SIGMA std, derived from standard normal distribution * SIGMA
x = np.random.randn(1, NBINS) * SIGMA

np.savez('obs_10000.npz', x=x)

x = np.random.randn(NBINS) * SIGMA
np.save('obs_10000_clean.npy', x)
