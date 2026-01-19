# DeployedGraph

The `DeployedGraph` class orchestrates distributed execution of Falcon graphs using Ray.

## Overview

`DeployedGraph` wraps a `Graph` and handles:

- Ray actor initialization for each node
- Distributed sample generation and training
- Coordination between nodes during inference

## Class Reference

::: falcon.core.deployed_graph.DeployedGraph
    options:
      show_source: true
