import numpy as np

class Signal:
    def __init__(self, n_bins=10000):
        self.t = np.linspace(0, 10, n_bins)
        self.t = np.float32(self.t)

    def simulate(self, z):
        y = (z[0] + z[1]*self.t) * np.sin( (z[2] + z[3]*self.t) * self.t + z[4] )
        y = np.float32(y)
        return y
    
class Noise:
    def __init__(self, sigma=1.0, n_bins=10000):
        self.sigma = sigma
        self.n_bins = n_bins

    def simulate(self):
        noise = self.sigma * np.random.randn(self.n_bins)
        noise = np.float32(noise)
        return noise

class Data:
    def simulate(self, y, n):
        x = y + n
        return x
