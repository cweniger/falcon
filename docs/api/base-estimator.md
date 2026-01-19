# BaseEstimator

The abstract base class defining the estimator interface.

## Overview

All estimators in Falcon must implement this interface. The base class defines
methods for:

- Training (`train`)
- Sampling (`sample_prior`, `sample_posterior`, `sample_proposal`)
- Persistence (`save`, `load`)
- Control flow (`pause`, `resume`, `interrupt`)

## Class Reference

::: falcon.core.base_estimator.BaseEstimator
    options:
      show_source: true
      members:
        - train
        - sample_prior
        - sample_posterior
        - sample_proposal
        - save
        - load
        - pause
        - resume
        - interrupt
