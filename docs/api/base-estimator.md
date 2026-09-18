# BaseEstimator

The model contract that every estimator implements, independent of the
network framework.

## Overview

An estimator node runs the same model class in two Ray actors:

- the **train actor** (`<node>/train`) drives it through
  `falcon.core.round_trainer.RoundTrainer`, which owns the round loop, the
  round counters, the best state and the checkpoint;
- the **sample actor** (`<node>`) drives it through
  `falcon.core.model_sampler.ModelSampler`, which installs the best state and
  falls back to the prior until one exists.

The model keeps its networks as mutable state on `self` and knows nothing
about rounds, publishing or Ray. The two sides exchange the best state as a
*state tree*: a nested dict of numpy arrays and plain values
(`falcon.core.state_io`). The same tree is written to
`graph/<node>/best_state.npz`.

Methods, grouped by who calls them:

| Called by | Methods |
|-----------|---------|
| both | `setup`, `groups`, `built`, `build`, `import_state` |
| train actor | `init_from_batch`, `train_step`, `evaluate`, `discard_mask`, `on_round_start`, `on_validation_end`, `snapshot`, `restore`, `export_state` |
| sample actor | `sample_prior`, `sample` |

Rules that let any framework implement the contract:

- `snapshot()` / `restore()` are fast in-memory copies with opaque handles;
- `export_state()` returns copies as numpy; nothing outside the model keeps
  references to the model's own arrays;
- sampling draws its randomness from the `rng` argument.

Torch estimators derive from `falcon.estimators.TorchModel`, which implements
the state handling for `nn.Module` network groups; `Flow` and
`GaussianFullCov` are examples.

## Writing a JAX estimator

JAX functions are pure, so a JAX model keeps its latest state on `self` and
replaces it after each compiled step. With buffer donation the step reuses the
old buffers, so memory does not double; old arrays then become invalid, and
`snapshot()` must copy:

```python
import jax, jax.numpy as jnp, numpy as np, optax
from falcon.core.base_estimator import BaseEstimator

class JaxGaussian(BaseEstimator):
    def build(self, init_tree):
        self.params = init_params(init_tree)              # pytree on the device
        self.opt = optax.adamw(self.lr)
        self.opt_state = self.opt.init(self.params)
        self._step = jax.jit(make_step(self.opt), donate_argnums=(0, 1))

    def groups(self):
        return {"model": "loss"}

    @property
    def built(self):
        return hasattr(self, "params")

    def train_step(self, batch):
        batch = {k: jnp.asarray(v.cpu().numpy()) for k, v in batch.items()}
        self.params, self.opt_state, loss = self._step(self.params, self.opt_state, batch)
        return {"loss": float(loss)}

    def snapshot(self, group):
        return jax.tree.map(jnp.copy, self.params)       # donation invalidates old arrays

    def restore(self, group, handle):
        self.params = jax.tree.map(jnp.copy, handle)

    def export_state(self, group):
        return {"params": flatten_to_numpy(self.params)}   # nested dict of np.ndarray

    def import_state(self, group, tree):
        self.params = unflatten_from_numpy(tree["params"])

    def sample(self, rng, num_samples, conditions, mode):
        key = jax.random.key(int(rng.integers(2**63 - 1)))
        ...
```

Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` through the node's
`ray.runtime_env.env_vars` when a JAX actor shares its GPU with other actors;
JAX otherwise reserves most of the GPU memory at start-up.

## Class Reference

::: falcon.core.base_estimator.BaseEstimator
    options:
      show_source: true

::: falcon.core.base_estimator.RoundConfig

::: falcon.estimators.torch_model.TorchModel
    options:
      show_source: false
