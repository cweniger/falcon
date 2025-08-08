import time
import zarr
import ray
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from sbi.analysis import pairplot
from pathlib import Path

import falcon
from falcon.contrib.norms import LazyOnlineNorm, DiagonalWhitener
from falcon.contrib.svd import PCAProjector

SIGMA = 1e-1
MU = 0.0
NPAR = 4
NBINS = 100


class Signal:
    def __init__(self):
        pass

    def sample(self, num_samples, parent_conditions=[]):
        z = parent_conditions[0]
        z = torch.tensor(z)
        y = torch.linspace(-torch.pi, torch.pi, NBINS).double()
        T = torch.stack([torch.cos(y*5), torch.sin(y*5), torch.cos(30*y), torch.sin(30*y)])
        m = z @ T
        return m.numpy()

    def get_shape_and_dtype(self):
        return (NBINS,), 'float64'


class Noise:
    def sample(self, num_samples, parent_conditions=[]):
        result = torch.randn((num_samples, NBINS)).double()*SIGMA
        return result.numpy()

    def get_shape_and_dtype(self):
        return (NBINS,), 'float64'


class Add:
    def sample(self, num_samples, parent_conditions=[]):
        m, n = parent_conditions
        return m + n

    def get_shape_and_dtype(self):
        return (NBINS,), 'float64'


class E(torch.nn.Module):
    def __init__(self):
        super(E, self).__init__()
        self.linear = torch.nn.LazyLinear(NPAR*2)
        self.projector = PCAProjector(buffer_size = 128)
        self.whitener = DiagonalWhitener(NBINS, use_fourier=False)

    def forward(self, x, *args):
        if len(args) > 0:  # Scaffolds provided
            m, n = args
            self.whitener.update(n)
            white_m = self.whitener(m)
            self.projector.update(white_m)
        x = self.projector(x).float()
        x = self.linear(x)
        return x


#async def main():
def main():
    ### User defined code

    num_epochs = 40
    #n_train = 4096
    n_train = 1024
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    filepath = Path('../runs/test_'+current_time)
    #filepath = '/scratch-local/weniger/zarr_stores/test_'+current_time+'.zarr'

    # p(z) and q(z|x)
    priors = [['uniform', MU-10*SIGMA, MU+10*SIGMA]]*NPAR
    node_z = falcon.Node("z",
                "falcon.contrib.SNPE_A",
                parents=[], evidence=['x',], scaffolds = ['m', 'n'],
                module_config=dict(embeddings = E, priors = priors, device='cpu', num_epochs = num_epochs, discard_samples=True, gamma = 0.3,
                    lr_decay_factor=0.1, lr=1e-2),   # 0.1, 0.5, 1.0
                actor_config=dict(num_gpus=0)
                )  

    # p(x|z)
    node_x = falcon.Node("x", Add, parents=['m', 'n'], observed=True, resample=True)
    node_m = falcon.Node("m", Signal, parents=['z'])
    node_n = falcon.Node("n", Noise)

    graph = falcon.Graph([node_z, node_x, node_m, node_n])
    print(graph)


    #########################
    ### Boiler plate code ###
    #########################

    # 0) Deploy graph
    deployed_graph = falcon.DeployedGraph(graph)

    # 1) Prepare dataset manager for deployed graph and store initial samples
    shapes_and_dtypes = deployed_graph.get_shapes_and_dtypes()
    dataset_manager = falcon.get_zarr_dataset_manager(shapes_and_dtypes, filepath,
            num_min_sims = n_train, num_val_sims=128, num_resims = 512)

    dataset_manager.generate_samples(deployed_graph, num_sims = 1024)

    conditions = dict(z = torch.zeros(1, 4).double())
    sample = deployed_graph.sample(1, conditions = conditions)
    observations = dict(x = torch.tensor(sample['x']))

    try:
        deployed_graph.launch(dataset_manager, observations)
    except KeyboardInterrupt:
        pass

    # 4) Evaluation and storage (here sample from the trained graph)
    samples = deployed_graph.conditioned_sample(1000, observations)
    #plot_samples = torch.stack([samples['z'][:,0],  samples['z'][:,1]]).T
    plot_samples = samples['z']
    #pairplot(plot_samples, limits=[[-5*SIGMA, 5*SIGMA]]*NPAR, figsize=(10, 10))
    pairplot((plot_samples-MU)/SIGMA, limits=[[-6, 6]]*NPAR, figsize=(15, 15))
    std_samples = torch.tensor(plot_samples).std(axis=0)
    print(std_samples)
    plt.savefig("../runs/figures/test_"+current_time+".png")
    plt.show()

    # 5) Clean up Ray resources
    deployed_graph.shutdown()

if __name__ == "__main__":
    main()
