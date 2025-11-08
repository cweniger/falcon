import numpy as np

class Signal:
    def __init__(self, n_bins=10000):
        self.t = np.linspace(0, 10, n_bins)

    def simulate(self, z):
        y = (z[0] + z[1]*self.t) * np.sin( (z[2] + z[3]*self.t) * self.t + z[4] )
        return y
    
class Noise:
    def __init__(self, sigma=1.0, n_bins=10000):
        self.sigma = sigma
        self.n_bins = n_bins

    def simulate(self):
        noise = self.sigma * np.random.randn(self.n_bins)
        return noise

class Data:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def simulate(self, y, n):
        x = y + n
        return x