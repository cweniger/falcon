# Graph

The graph module provides the core data structures for defining probabilistic models.

## Overview

A Falcon graph consists of `Node` objects connected by parent-child relationships.
The `Graph` class manages these relationships and handles topological sorting for
correct execution order.

## Classes

::: falcon.core.graph.Node
    options:
      members:
        - __init__

::: falcon.core.graph.Graph
    options:
      members:
        - __init__
        - get_parents
        - get_evidence
        - get_simulator_cls

## Functions

::: falcon.core.graph.CompositeNode

::: falcon.core.graph.create_graph_from_config
