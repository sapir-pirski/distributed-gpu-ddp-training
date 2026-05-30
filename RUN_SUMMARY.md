# Run Summary

Source log: `training_log.txt`

## Distributed Setup

| Field | Value |
| --- | --- |
| Launcher | `torchrun` |
| Nodes | 2 |
| GPUs per node | 1 |
| World size | 2 |
| Node ranks | 0, 1 |
| Backend | `nccl` |
| Master address | `10.4.54.78` |
| Master port | `29500` |

## Training Configuration

| Field | Value |
| --- | --- |
| Model | `facebook/opt-1.3b` |
| Max steps | 10 |
| Accelerator request | `H200:1` |
| Docker image | `docker:cr.eu-north1.nebius.cloud/e00avpz7r2gn4zffdk/nebius-trainer:v1` |

## Training Result

| Field | Value |
| --- | --- |
| Status | SUCCEEDED |
| Train runtime | 37.76 sec |
| Train samples/sec | 2.119 |
| Train steps/sec | 0.265 |
| Train loss | 8.061 |
| Eval loss | 8.81 |
| Eval runtime | 1.707 sec |
| Eval samples/sec | 295.3 |
| Epoch | 0.01678 |

## Evidence

- `torchrun` launched on node ranks 0, 1.
- NCCL initialized with world size 2 and backend `nccl`.
- Training completed successfully for 10 steps.
- NCCL communicators were destroyed cleanly.
- SkyPilot reported `Job finished (status: SUCCEEDED)`.

## Notes

- This run validates the distributed training infrastructure rather than model quality.
- The assignment submission zip intentionally excludes this summary because `TASK.md` requires exactly five files.
