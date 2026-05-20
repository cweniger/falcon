# Falcon: Noise Resampling via Story Scaffolds

## Motivation

In SBI problems with a factored forward model

```
theta → signal → x = signal + noise
               ↑
             noise
```

the signal simulation is expensive while the noise draw is cheap.  Today Falcon
simulates the full chain once and caches everything, so each expensive signal is
paired with exactly one noise realization during training.

With noise resampling, each cached signal is paired with a **permuted** noise draw
from another sample in the batch on every training step — no simulator calls at
training time, no parametric noise assumption required.  The cached noise realizations
themselves serve as the noise library.

**Effective sample count:**
```
1 000 signal simulations × N noise permutations/step ≈ N 000 effective (theta, x) pairs
```

This is valid for SNPE-C: importance weights depend only on theta, not on x, so
resampling x from p(x | signal) is exact.  The permutation trick additionally works
for any noise distribution — Gaussian, Poisson, correlated instrument noise — without
needing to specify a parametric noise model at training time.

---

## Core Design: Graph Node and Embedding Wrapper Are Mirrors

The key insight is that the graph node computing `x = signal + noise` and the
embedding wrapper computing `x = signal + noise[perm]` are the same operation —
one at simulation time, one at training time.  This symmetry drives the design.

| | Graph: `x` node | Embedding: `NoisePermutation` |
|---|---|---|
| Inputs | signal, noise | signal, noise |
| Output | `signal + noise` | `signal + noise[perm]` |
| When | simulation time | each training step |
| At inference | — | passes `x` through unchanged |

Both are thin wrappers around the same additive structure.  The graph expresses the
forward model; the embedding wrapper replays it with fresh noise.

---

## Graph Structure

No `CompositeNode` needed.  Signal and noise are separate first-class nodes:

```yaml
graph:
  theta:
    evidence: [x]
    scaffolds: [signal, noise]           # carry the full story into training
    estimator:
      _target_: falcon.estimators.Flow
      embedding:
        _target_: falcon.embeddings.NoisePermutation
        _input_: [x, signal, noise]
        inner:
          _target_: model.MyEmbedding
          _input_: [x, signal, noise]    # or just [x] for simpler architectures

  signal:
    parents: [theta]
    simulator:
      _target_: model.SignalSimulator    # expensive

  noise:
    simulator:
      _target_: model.NoiseSimulator    # cheap — no parents needed for iid noise
                                        # can have parents for heteroscedastic noise

  x:
    parents: [signal, noise]
    simulator:
      _target_: falcon.simulators.Sum   # x = signal + noise  (new utility, ~3 lines)
    observed: "./data/obs.npz['x']"
```

The graph reads as a direct description of the forward model.  `signal` and `noise`
are scaffolds, which already flow into `condition_keys` and are already fetched by
the `CachedDataLoader` — zero infrastructure changes needed.

---

## Simulator Convention

Signal and noise simulators are independent:

```python
class SignalSimulator:
    """Expensive part: theta → signal."""
    def simulate_batch(self, batch_size, theta):
        return (theta @ self.Phi.T).cpu().numpy()

class NoiseSimulator:
    """Cheap part: () → noise."""
    def simulate_batch(self, batch_size):
        return np.random.randn(batch_size, self.n_bins) * self.sigma
```

No dict return, no special interface.  Standard `simulate_batch` throughout.

---

## New Falcon Primitives (~35 lines total)

### `falcon/simulators/sum.py`

```python
class Sum:
    """Simulator that adds its parent values: x = sum(parents)."""
    def simulate_batch(self, batch_size, *inputs):
        result = inputs[0].copy()
        for inp in inputs[1:]:
            result = result + inp
        return result
```

Mirrors `falcon.core.Extractor` in simplicity.  Also useful beyond noise resampling
(any additive composition in a graph).

### `falcon/embeddings/noise_permutation.py`

```python
class NoisePermutation(nn.Module):
    """Embedding wrapper that permutes noise across the batch during training.

    Training (signal and noise are tensors):
        perm      = randperm(batch_size)
        noise_new = noise[perm]
        x_new     = signal + noise_new
        → inner(x_new, signal, noise_new)

    Inference (signal and noise are None — embedding builder passes None
    for missing scaffold keys automatically):
        → inner(x, None, None)

    The inner embedding sees the same interface in both modes.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x, signal=None, noise=None):
        if signal is not None and noise is not None:
            perm  = torch.randperm(x.shape[0], device=x.device)
            noise = noise[perm]
            x     = signal + noise
        return self.inner(x, signal, noise)
```

### Registration

Add both to `falcon/simulators/__init__.py` and `falcon/embeddings/__init__.py`.

---

## User Embedding Patterns

**Minimal** — only benefits from noise augmentation, ignores auxiliary inputs:

```python
class MyEmbedding(nn.Module):
    def forward(self, x, signal=None, noise=None):
        return self.net(x)
```

**Full story** — uses signal as an auxiliary branch to help disentangle:

```python
class StoryEmbedding(nn.Module):
    def forward(self, x, signal=None, noise=None):
        h = self.embed_x(x)
        if signal is not None:
            h = h + self.embed_signal(signal)
        return h
```

Both work at inference without any mode flag — `signal` and `noise` are simply
`None` because scaffold keys aren't present in the inference conditions dict.

---

## Why Permutation (Not Fresh Sampling)

| | Permutation | Fresh sampling |
|---|---|---|
| Noise model at training time | Not needed | Must be implemented |
| Non-Gaussian / correlated noise | Works automatically | Must be modelled |
| Simulator calls per training step | Zero | One per batch |
| Noise realizations are realistic | Yes — from actual simulations | May differ from simulated |

For LISA (complex instrument noise PSD) or any problem where the noise distribution
is hard to parametrise, permutation automatically uses real noise draws.

The one requirement: buffer must contain at least `batch_size` noise samples.
Always satisfied in practice.

---

## Interaction with Importance Weights

Falcon's Flow estimator (SNPE-C) computes importance weights as
`p(theta) / q(theta)`.  These depend only on theta.  Permuting noise changes which
`x` is paired with each theta, but each new pair `(theta_i, signal_i + noise_j)` is
a valid draw from the joint `p(theta, x)` under i.i.d. noise.  No weight correction
needed.

---

## What Changes in Falcon

### New files

- `falcon/simulators/sum.py` — `Sum` simulator (~8 lines)
- `falcon/embeddings/noise_permutation.py` — `NoisePermutation` wrapper (~20 lines)

### Modified files

- `falcon/simulators/__init__.py` — export `Sum`
- `falcon/embeddings/__init__.py` — export `NoisePermutation`
- `examples/05_linear_regression/src/model.py` — split `LinearSimulator` into
  `SignalSimulator` + `NoiseSimulator`
- `examples/05_linear_regression/config.yaml` — add `signal`, `noise`, `x` nodes;
  add scaffolds; add `NoisePermutation` wrapper

### No changes to

- `falcon/core/raystore.py` — scaffolds already flow through `CachedDataLoader`
- `falcon/core/deployed_graph.py` — scaffold wiring already exists
- `falcon/core/graph.py` — `scaffolds` field and parsing already exist
- `falcon/estimators/flow.py` — already passes full `conditions` dict to embedding
- `falcon/embeddings/builder.py` — already handles missing keys as `None`,
  already resolves nested `inner:` configs

---

## Validation

1. Run `examples/05_linear_regression` with and without scaffolds + `NoisePermutation`.
   With 10× fewer simulations and noise resampling, posterior quality should match
   the full-simulation baseline.

2. Confirm that at inference, the embedding receives `signal=None, noise=None` and
   falls back to real `x` without warnings or errors.

3. With a `StoryEmbedding` using the auxiliary signal branch, confirm gradients
   flow through all inputs during training.

---

## Estimated Effort

~35 lines of new Falcon code + example update.  No changes to any core
infrastructure.  Depends on `PLAN_CACHED_DATALOADER.md` for efficient scaffold
key fetching via the cached loader.
