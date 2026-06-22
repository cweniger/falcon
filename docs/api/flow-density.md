# FlowDensity

Normalizing flow networks for density estimation.

## Overview

`FlowDensity` is the internal `nn.Module` that wraps various normalizing flow
architectures for use in posterior estimation. It is used internally by the
[Flow](flow.md) estimator.

## Supported Flow Types

`zuko_*` types are available by default. Types backed by `sbi/nflows` require the
optional `sbi` extra: `pip install falcon-sbi[sbi]`.

| Type | Library | Description |
|------|---------|-------------|
| `nsf` | sbi/nflows | Neural Spline Flow (recommended when `sbi` is installed) |
| `maf` | sbi/nflows | Masked Autoregressive Flow |
| `made` | sbi | Masked Autoencoder for Distribution Estimation |
| `maf_rqs` | sbi | MAF with Rational Quadratic Splines |
| `zuko_nice` | Zuko | NICE architecture (default — no extra required) |
| `zuko_maf` | Zuko | Masked Autoregressive Flow (Zuko) |
| `zuko_nsf` | Zuko | Neural Spline Flow (Zuko) |
| `zuko_ncsf` | Zuko | Neural Circular Spline Flow |
| `zuko_sospf` | Zuko | Sum-of-Squares Polynomial Flow |
| `zuko_naf` | Zuko | Neural Autoregressive Flow |
| `zuko_unaf` | Zuko | Unconstrained NAF |
| `zuko_gf` | Zuko | Glow Flow |
| `zuko_bpf` | Zuko | Bernstein Polynomial Flow |

## Class Reference

::: falcon.estimators.flow_density.FlowDensity
    options:
      show_source: true
