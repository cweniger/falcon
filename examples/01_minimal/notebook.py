# %% [markdown]
# # 01 — Minimal Falcon run (notebook API)
#
# This notebook shows the simplest way to run Falcon from Python/Colab:
# load a config, optionally tweak a parameter, launch training, and inspect
# the result.  The matching CLI command is:
#
# ```bash
# cd examples/01_minimal
# falcon launch -o output/my_run
# ```
#
# **Prerequisites**: install Falcon and its dependencies, then run this
# notebook from the `examples/01_minimal/` directory so that the relative
# paths in `config.yml` resolve correctly.

# %% [markdown]
# ## 1. Load the config

# %%
import falcon

cfg = falcon.config("config.yml")
cfg  # rich repr renders the full YAML in Jupyter

# %% [markdown]
# ## 2. Override parameters for a quick demo run
#
# `override()` returns a new `Config`; the original is unchanged.
# Use dotted paths matching the YAML structure.

# %%
cfg = cfg.override(
    "buffer.min_samples=256",
    "buffer.max_samples=1024",
    "buffer.validation_samples=64",
    "graph.z.estimator.loop.max_epochs=5",
    "graph.z.estimator.loop.early_stop_patience=5",
    "sample.posterior.n=200",
)

# %% [markdown]
# ## 3. Launch training
#
# `falcon.launch()` blocks until training completes and returns a `Run`
# object pointing at the output directory.  Ray is started automatically
# on the first call if it is not already running.

# %%
run = falcon.launch(cfg, output="output/notebook_run")
run

# %% [markdown]
# ## 4. Inspect the result

# %%
# Path where everything was written
print("Output dir:", run.run_dir)

# Loaded config (identical to what was saved at the start of the run)
print("\nConfig keys:", list(run.config.keys()))

# Posterior samples (written by auto_sample=True)
samples = run.samples
print("\nSamples:", samples)
