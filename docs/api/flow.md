# Flow

Normalizing flow networks for density estimation.

## Overview

The `Flow` class wraps various normalizing flow architectures for use in
posterior estimation. It supports multiple flow types from different libraries.

## Supported Flow Types

| Type | Library | Description |
|------|---------|-------------|
| `nsf` | sbi/nflows | Neural Spline Flow |
| `maf` | sbi/nflows | Masked Autoregressive Flow |
| `zuko_nice` | Zuko | NICE architecture |
| `zuko_naf` | Zuko | Neural Autoregressive Flow |
| `zuko_nsf` | Zuko | Neural Spline Flow (Zuko) |
| `zuko_maf` | Zuko | Masked Autoregressive Flow (Zuko) |

## Class Reference

::: falcon.contrib.flow.Flow
    options:
      show_source: true
