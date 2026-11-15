# Project rules

## Research decisions
- Do not take autonomous decisions about the research project (methods, losses, hyperparameters, augmentations, protocol, datasets, experiment design).
- Discuss every important decision with the user and get agreement **before** implementing or running anything. When a choice comes up during implementation, stop and ask rather than picking a default.

## Shared workstation: GPU usage
- This workstation (4x RTX 3090) is shared with other users.
- Use **at most 2 GPUs at the same time**, in total across all jobs (training, benchmarks, tests). Always leave at least 2 GPUs free.
- Always set `CUDA_VISIBLE_DEVICES` explicitly for every GPU job. Before launching, check with `nvidia-smi` which GPUs are free and pick from those.
