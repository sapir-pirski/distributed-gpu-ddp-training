#!/usr/bin/env python3
"""Validate the local assignment artifacts without contacting cloud services."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import yaml


REQUIRED_SUBMISSION_FILES = [
    "mk8s-ng-config.json",
    "Dockerfile",
    "train.py",
    "train_job.yaml",
    "training_log.txt",
]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_files() -> None:
    missing = [name for name in REQUIRED_SUBMISSION_FILES if not Path(name).is_file()]
    if missing:
        fail(f"missing required submission files: {', '.join(missing)}")


def validate_node_group_config() -> None:
    config = json.loads(Path("mk8s-ng-config.json").read_text(encoding="utf-8"))
    if sorted(config.keys()) != ["metadata", "spec"]:
        fail("mk8s-ng-config.json must contain exactly metadata and spec")

    metadata = config["metadata"]
    spec = config["spec"]
    if not metadata.get("id") or not metadata.get("name"):
        fail("mk8s-ng-config.json metadata must include id and name")

    fixed_count = str(spec.get("fixed_node_count", ""))
    if fixed_count != "2":
        fail("mk8s-ng-config.json spec.fixed_node_count must be 2")

    resources = spec.get("template", {}).get("resources", {})
    if not resources.get("platform") or not resources.get("preset"):
        fail("mk8s-ng-config.json must include template.resources platform and preset")


def validate_train_job() -> None:
    job = yaml.safe_load(Path("train_job.yaml").read_text(encoding="utf-8"))
    if job.get("num_nodes") != 2:
        fail("train_job.yaml must set num_nodes: 2")

    resources = job.get("resources", {})
    if not str(resources.get("infra", "")).startswith("k8s/"):
        fail("train_job.yaml resources.infra must target Kubernetes")

    if not resources.get("accelerators"):
        fail("train_job.yaml must request a GPU accelerator")

    image_id = str(resources.get("image_id", ""))
    if not image_id.startswith("docker:"):
        fail("train_job.yaml resources.image_id must point to a Docker image")

    envs = job.get("envs", {})
    if envs.get("NCCL_DEBUG") != "INFO":
        fail("train_job.yaml must set NCCL_DEBUG: INFO")

    run_block = str(job.get("run", ""))
    for expected in ("torchrun", "--nnodes", "--node_rank", "--master_addr"):
        if expected not in run_block:
            fail(f"train_job.yaml run block is missing {expected}")


def validate_training_log() -> None:
    log = Path("training_log.txt").read_text(encoding="utf-8", errors="replace")
    for expected in (
        "torchrun",
        "[NCCL] Distributed training initialised",
        "[NCCL] World size: 2",
        "[NCCL] Backend: nccl",
        "Training complete",
        "Job finished (status: SUCCEEDED)",
    ):
        if expected not in log:
            fail(f"training_log.txt is missing expected evidence: {expected}")


def validate_zip(zip_path: Path) -> None:
    if not zip_path.is_file():
        fail(f"{zip_path} does not exist")

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()

    if names != REQUIRED_SUBMISSION_FILES:
        fail(
            "submission zip must contain exactly "
            f"{REQUIRED_SUBMISSION_FILES}, got {names}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        type=Path,
        help="Optional submission zip to validate for exact file contents.",
    )
    args = parser.parse_args()

    require_files()
    validate_node_group_config()
    validate_train_job()
    validate_training_log()
    if args.zip:
        validate_zip(args.zip)

    print("Project validation passed.")


if __name__ == "__main__":
    main()
