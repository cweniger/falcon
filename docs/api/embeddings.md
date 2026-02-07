# Embeddings

Declarative embedding pipelines for observation processing.

## Overview

The `falcon.embeddings` package provides tools for building observation embedding
networks via declarative YAML configuration. Embeddings map raw observations to
lower-dimensional summary statistics before they enter the estimator.

## Declarative Configuration

Embedding networks are defined in YAML using the `_target_` / `_input_` system:

- **`_target_`**: Import path of the `nn.Module` class to instantiate
- **`_input_`**: List of observation node names (or nested sub-embeddings) that feed into this module

### Basic Embedding

```yaml
embedding:
  _target_: model.MyEmbedding
  _input_: [x]
```

### Multi-Input Embedding

```yaml
embedding:
  _target_: model.MyEmbedding
  _input_: [x, y]  # Multiple observation nodes
```

### Nested Pipeline

Sub-embeddings can be nested arbitrarily deep. Each `_input_` entry can itself
be a `_target_` / `_input_` block:

```yaml
embedding:
  _target_: model.Concatenate
  _input_:
    - _target_: timm.create_model
      model_name: resnet18
      pretrained: true
      num_classes: 0
      _input_:
        _target_: model.Unsqueeze
        _input_: [image]
    - _target_: torch.nn.Linear
      in_features: 64
      out_features: 32
      _input_: [metadata]
```

## Built-in Utilities

### Normalization

#### LazyOnlineNorm

Online normalization with lazy initialization and optional momentum adaptation.
Normalizes inputs to zero mean and unit variance using exponential moving
averages.

```yaml
embedding:
  _target_: falcon.embeddings.LazyOnlineNorm
  _input_: [x]
  momentum: 0.01
```

#### DiagonalWhitener

Diagonal whitening with optional Hartley transform preprocessing and
momentum-based running statistics.

```yaml
embedding:
  _target_: falcon.embeddings.DiagonalWhitener
  _input_: [x]
```

#### hartley_transform

A standalone function for computing the discrete Hartley transform, useful as a
preprocessing step.

### Dimensionality Reduction

#### PCAProjector

Streaming dual PCA projector with momentum-based updates. Performs online SVD for
dimensionality reduction with variance-based prior (ridge-like regularization)
and optional output normalization.

```yaml
embedding:
  _target_: falcon.embeddings.PCAProjector
  _input_: [x]
  n_components: 32
```

## Class Reference

::: falcon.embeddings.builder.instantiate_embedding

::: falcon.embeddings.builder.EmbeddingWrapper
    options:
      show_source: true

::: falcon.embeddings.norms.LazyOnlineNorm
    options:
      show_source: true

::: falcon.embeddings.norms.DiagonalWhitener
    options:
      show_source: true

::: falcon.embeddings.norms.hartley_transform

::: falcon.embeddings.svd.PCAProjector
    options:
      show_source: true
