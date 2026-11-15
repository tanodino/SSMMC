# Project rules

## Shared workstation: GPU usage
- This workstation (4x RTX 3090) is shared with other users.
- Use **at most 2 GPUs at the same time**, in total across all jobs (training, benchmarks, tests). Always leave at least 2 GPUs free.
- Always set `CUDA_VISIBLE_DEVICES` explicitly for every GPU job. Before launching, check with `nvidia-smi` which GPUs are free and pick from those.
