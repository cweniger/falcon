# %% [markdown]
# # 04 — Gaussian posterior: programmatic graph API
#
# This notebook demonstrates the **Python-first** way to define a Falcon model:
# build the graph with `Graph.add_node()` instead of a YAML config file.
# The forward model and embedding are plain Python callables defined right here
# in the notebook — no separate `src/model.py` needed.
#
# The same model is also runnable via the CLI:
# ```bash
# cd examples/04_gaussian
# falcon launch -o output/cli_run
# ```
# That CLI path uses `config.yml` + `src/model.py`.  Both paths produce
# identical results; the notebook path is the "define your own" lesson.
#
# **Prerequisites**: run `python data/gen_mock_data.py` once to create the
# mock observation, then execute this notebook from `examples/04_gaussian/`.

# %% [markdown]
# ## 1. Define the forward model and embedding in Python

# %%
import numpy as np
import torch
import torch.nn as nn
import falcon


class ExpPlusNoise:
    """Forward model: x = exp(z) + noise.  Plain callable, no base class needed."""

    def __init__(self, sigma: float = 1e-6):
        self.sigma = sigma

    def simulate_batch(self, batch_size, z):
        z = torch.tensor(z)
        x = torch.exp(z) + torch.randn_like(z) * self.sigma
        return x.numpy()


class IdentityEmbedding(nn.Module):
    """Pass-through embedding: observation x is fed directly to the network."""

    def forward(self, inputs: dict) -> torch.Tensor:
        return inputs["x"]


# %% [markdown]
# ## 2. Load the observation

# %%
obs = np.load("data/mock_data.npz")["x"]  # shape (3,)
print("Observation shape:", obs.shape, " values:", obs)

# %% [markdown]
# ## 3. Build the graph programmatically
#
# `falcon.Graph()` starts empty.  `add_node()` accepts live Python objects
# for `simulator=` and `estimator=`; they are shipped to Ray actors via
# cloudpickle — no importable path required.

# %%
graph = falcon.Graph()

graph.add_node(
    "z",
    simulator=falcon.priors.Product([
        ["normal", 0.0, 1.0],
        ["normal", 0.0, 1.0],
        ["normal", 0.0, 1.0],
    ]),
    estimator=falcon.estimators.GaussianFullCov,   # class: instantiated by the graph
    evidence=["x"],
    ray_num_gpus=0,
)

graph.add_node(
    "x",
    simulator=ExpPlusNoise(sigma=1e-6),            # live instance via cloudpickle
    parents=["z"],
    observed=obs,                                  # ndarray passed directly
    ray_num_gpus=0,
)

graph  # shows ASCII graph repr

# %% [markdown]
# ## 4. Launch training
#
# `falcon.launch(graph)` synthesises a default config (buffer, paths, logging),
# saves it as `config.yml` in the output directory, and runs training.
# Pass `overrides=` to customise buffer size or epoch count.

# %%
run = falcon.launch(
    graph,
    output="output/notebook_run",
    overrides=[
        "buffer.min_samples=512",
        "buffer.max_samples=1024",
        "buffer.validation_samples=128",
        "sample.posterior.n=200",
    ],
)
run

# %% [markdown]
# ## 5. Inspect the saved config
#
# The saved `config.yml` is human-readable; live Python objects appear as
# `<live object: ClassName>` placeholders so the file is honest about
# reproducibility.

# %%
cfg_path = run.run_dir / "config.yml"
print(cfg_path.read_text())
