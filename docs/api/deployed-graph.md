# DeployedGraph

The `DeployedGraph` class orchestrates distributed execution of Falcon graphs using Ray.

## Overview

`DeployedGraph` wraps a `Graph` and handles:

- one or more sample actors per node (`SampleActor`, named after the node),
  which serve prior, proposal and posterior samples;
- one train actor per estimator node (`TrainActor`, named `<node>/train`),
  which trains the networks and publishes each new best state to the node's
  sample actors;
- the simulation loop that fills the buffer while the train actors run.

Pass `train=False` to deploy sample actors only, as `falcon sample` does.

## Class Reference

::: falcon.core.deployed_graph.DeployedGraph
    options:
      show_source: true

::: falcon.core.deployed_graph.resolve_actor_options
