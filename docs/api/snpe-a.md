# SNPE_A

Sequential Neural Posterior Estimation (SNPE-A) implementation.

## Overview

`SNPE_A` is the primary estimator in Falcon for learning posterior distributions.
It uses dual normalizing flows (conditional and marginal) with importance sampling
for adaptive proposal generation.

Key features:

- Dual flow architecture for posterior and proposal sampling
- Parameter space normalization via hypercube mapping
- Importance sampling with effective sample size monitoring
- Automatic learning rate scheduling and early stopping

## Configuration

SNPE_A is configured via nested dataclasses:

```yaml
estimator:
  _target_: falcon.contrib.SNPE_A
  loop:
    num_epochs: 300
    batch_size: 128
  network:
    net_type: nsf
  optimizer:
    lr: 0.01
  inference:
    gamma: 0.5
```

## Class Reference

::: falcon.contrib.SNPE_A.SNPE_A
    options:
      show_source: true
      members:
        - __init__
        - train_step
        - val_step
        - sample_prior
        - sample_posterior
        - sample_proposal
        - save
        - load

## Configuration Classes

::: falcon.contrib.SNPE_A.SNPEConfig

::: falcon.contrib.SNPE_A.NetworkConfig

::: falcon.contrib.SNPE_A.OptimizerConfig

::: falcon.contrib.SNPE_A.InferenceConfig

::: falcon.contrib.stepwise_estimator.TrainingLoopConfig
