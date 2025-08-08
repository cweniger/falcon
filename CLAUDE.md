# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FALCON (Federated Adaptive Learning of CONditional distributions) is a Python package for probabilistic graphical models using distributed computing with Ray. The package implements a graph-based framework for building complex probabilistic models where nodes represent distributions and edges represent dependencies.

## Installation and Setup

Install the development version:
```bash
pip install -e .
```

Dependencies are managed through setup.py and include:
- wandb>=0.15.0 (experiment tracking)
- torch>=2.0.0 (neural networks)
- zarr (data storage)
- numpy
- ray (distributed computing)
- sbi (simulation-based inference)

## Architecture

### Core Components

**falcon.core.graph**: Defines the conceptual graph structure
- `Node`: Represents a probabilistic node with parents, evidence, and configuration
- `Graph`: Container for nodes with topological sorting and dependency management
- `CompositeNode`: Helper for creating nodes with multiple outputs

**falcon.core.deployed_graph**: Handles distributed execution
- `DeployedGraph`: Deploys nodes as Ray actors for parallel execution
- `NodeWrapper`: Ray actor wrapper for individual nodes
- `MultiplexNodeWrapper`: Distributes computation across multiple actors

**falcon.core.zarrstore**: Data management using Zarr arrays
- `DatasetManagerActor`: Ray actor for managing simulation datasets
- `DatasetView`: Provides filtered views of stored data

**falcon.contrib**: Additional utilities and implementations
- Contains specialized priors, norms, and SBI implementations
- SNPE_A implementation for neural posterior estimation

### Key Concepts

1. **Graph Definition**: Models are defined as directed acyclic graphs where each node represents a distribution
2. **Forward Sampling**: Generates samples following parent-child dependencies
3. **Inference**: Uses evidence nodes to perform conditional sampling and posterior estimation
4. **Distributed Training**: Nodes can be trained independently using Ray actors
5. **Adaptive Simulation**: Dynamically generates new samples during training based on model requirements

## Development Workflow

Since this project uses a simple setup.py configuration, there are no predefined build, test, or lint commands. The package can be installed in development mode and imported directly.

## Code Patterns

- Nodes use lazy loading via `LazyLoader` to defer module instantiation until runtime
- Ray actors handle distributed computation with automatic resource management
- Zarr arrays provide efficient storage for large simulation datasets
- WandB integration for experiment tracking and logging
- Async/await patterns for coordinating distributed training

## Key Files to Understand

- `falcon/core/graph.py`: Core graph abstraction and topological sorting
- `falcon/core/deployed_graph.py`: Ray-based distributed execution (lines 191-380)
- `falcon/core/zarrstore.py`: Data storage and management
- `examples/`: Usage patterns and model definitions