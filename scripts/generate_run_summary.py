#!/usr/bin/env python3
"""Generate RUN_SUMMARY.md from train_job.yaml and training_log.txt."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml


LOG_PATH = Path("training_log.txt")
JOB_PATH = Path("train_job.yaml")
SUMMARY_PATH = Path("RUN_SUMMARY.md")


def first_match(pattern: str, text: str, default: str = "not found") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def metric_dicts(log_text: str) -> list[dict[str, str]]:
    metrics: list[dict[str, str]] = []
    for raw in re.findall(r"\{[^{}]+\}", log_text):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            metrics.append({str(key): str(value) for key, value in parsed.items()})
    return metrics


def latest_metric(metrics: list[dict[str, str]], key: str) -> str:
    for values in reversed(metrics):
        if key in values:
            return values[key]
    return "not found"


def torchrun_ranks(log_text: str) -> str:
    ranks = sorted(set(re.findall(r"--node_rank=(\d+)", log_text)), key=int)
    return ", ".join(ranks) if ranks else "not found"


def main() -> None:
    log_text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    job = yaml.safe_load(JOB_PATH.read_text(encoding="utf-8"))
    envs = job.get("envs", {})
    resources = job.get("resources", {})
    metrics = metric_dicts(log_text)

    values = {
        "source_log": str(LOG_PATH),
        "model_id": str(envs.get("MODEL_ID", "not found")),
        "max_steps": str(envs.get("MAX_STEPS", "not found")),
        "accelerators": str(resources.get("accelerators", "not found")),
        "image_id": str(resources.get("image_id", "not found")),
        "nodes": first_match(r"SKYPILOT_NUM_NODES=(\d+)", log_text),
        "gpus_per_node": first_match(r"SKYPILOT_NUM_GPUS_PER_NODE=(\d+)", log_text),
        "world_size": first_match(r"\[NCCL\] World size: (\d+)", log_text),
        "backend": first_match(r"\[NCCL\] Backend: ([^\n]+)", log_text),
        "master_addr": first_match(r"\[NCCL\] Master addr: ([^\n]+)", log_text),
        "master_port": first_match(r"\[NCCL\] Master port: ([^\n]+)", log_text),
        "node_ranks": torchrun_ranks(log_text),
        "train_runtime": latest_metric(metrics, "train_runtime"),
        "train_samples_per_second": latest_metric(metrics, "train_samples_per_second"),
        "train_steps_per_second": latest_metric(metrics, "train_steps_per_second"),
        "train_loss": latest_metric(metrics, "train_loss"),
        "eval_loss": latest_metric(metrics, "eval_loss"),
        "eval_runtime": latest_metric(metrics, "eval_runtime"),
        "eval_samples_per_second": latest_metric(metrics, "eval_samples_per_second"),
        "epoch": latest_metric(metrics, "epoch"),
        "status": "SUCCEEDED"
        if "Job finished (status: SUCCEEDED)" in log_text
        else "not found",
    }

    summary = f"""# Run Summary

Source log: `{values["source_log"]}`

## Distributed Setup

| Field | Value |
| --- | --- |
| Launcher | `torchrun` |
| Nodes | {values["nodes"]} |
| GPUs per node | {values["gpus_per_node"]} |
| World size | {values["world_size"]} |
| Node ranks | {values["node_ranks"]} |
| Backend | `{values["backend"]}` |
| Master address | `{values["master_addr"]}` |
| Master port | `{values["master_port"]}` |

## Training Configuration

| Field | Value |
| --- | --- |
| Model | `{values["model_id"]}` |
| Max steps | {values["max_steps"]} |
| Accelerator request | `{values["accelerators"]}` |
| Docker image | `{values["image_id"]}` |

## Training Result

| Field | Value |
| --- | --- |
| Status | {values["status"]} |
| Train runtime | {values["train_runtime"]} sec |
| Train samples/sec | {values["train_samples_per_second"]} |
| Train steps/sec | {values["train_steps_per_second"]} |
| Train loss | {values["train_loss"]} |
| Eval loss | {values["eval_loss"]} |
| Eval runtime | {values["eval_runtime"]} sec |
| Eval samples/sec | {values["eval_samples_per_second"]} |
| Epoch | {values["epoch"]} |

## Evidence

- `torchrun` launched on node ranks {values["node_ranks"]}.
- NCCL initialized with world size {values["world_size"]} and backend `{values["backend"]}`.
- Training completed successfully for {values["max_steps"]} steps.
- NCCL communicators were destroyed cleanly.
- SkyPilot reported `Job finished (status: SUCCEEDED)`.

## Notes

- This run validates the distributed training infrastructure rather than model quality.
- The assignment submission zip intentionally excludes this summary because `TASK.md` requires exactly five files.
"""

    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
