# DESIGN.md

- Simulation
  - Keywords
    - parent_keys
    - node_key
  - sample(*parent_args)

- Training
  - Keywords
    - node_key
    - parent_keys
    - embedding_keys 
    - scaffold_keys
  - dataloader
    - provides dict with node_key + parent_keys + embedding_keys
  - embedding(*embedding_args)

- Inference
  - posterior(*parent_args, *inference_keys)
    [inference_keys = embedding_keys - scaffold_keys]
  - 

- Proposition
