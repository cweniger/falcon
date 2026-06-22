# Hypercube (removed)

`Hypercube` has been removed. Use [`Product`](product.md) instead — it is a strict superset that also supports `mode="standard_normal"` (for the Gaussian estimator) and `"fixed"` parameters.

**Before:**
```yaml
simulator:
  _target_: falcon.priors.Hypercube
  priors:
    - ['uniform', -10.0, 10.0]
```

**After:**
```yaml
simulator:
  _target_: falcon.priors.Product
  priors:
    - ['uniform', -10.0, 10.0]
```

See the [Product reference](product.md) for the full API.
