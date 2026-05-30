#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-eu-north1}"
PROJECT_ID="${PROJECT_ID:-project-e00kqmm8pr00pmxqt43be3}"
REGISTRY_ID="${REGISTRY_ID:-e00avpz7r2gn4zffdk}"
REGISTRY_RESOURCE_ID="${REGISTRY_RESOURCE_ID:-registry-${REGISTRY_ID}}"
IMAGE_NAME="${IMAGE_NAME:-nebius-trainer}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
IMAGE_URI="${IMAGE_URI:-cr.${REGION}.nebius.cloud/${REGISTRY_ID}/${IMAGE_NAME}:${IMAGE_TAG}}"

CLUSTER_CONTEXT="${CLUSTER_CONTEXT:-nebius-ddp-mk8s}"
CLUSTER_ID="${CLUSTER_ID:-mk8scluster-e00gtp60n9mh1n1kva}"
NODE_GROUP_ID="${NODE_GROUP_ID:-mk8snodegroup-e00bcab74e3vwqwt50}"
SKY_CLUSTER="${SKY_CLUSTER:-ddp-run}"
SKY_JOB_ID="${SKY_JOB_ID:-}"
SUBMISSION_ZIP="${SUBMISSION_ZIP:-nebius-ddp-submission.zip}"

NEBIUS="${NEBIUS:-./.tools/nebius}"
SKY="${SKY:-.venv/bin/sky}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-$PWD/kubeconfig}"

export PATH="/opt/homebrew/opt/netcat/bin:$PWD/.tools:/opt/homebrew/bin:$PATH"
export KUBECONFIG="$KUBECONFIG_PATH"

print_steps() {
  cat <<STEPS
Step-by-step project run:

1. Create a Nebius Container Registry.
   Registry used here: cr.${REGION}.nebius.cloud/${REGISTRY_ID}

2. Configure Docker authentication for Nebius Registry.
   ${NEBIUS} registry configure-helper

3. Build the training Docker image.
   docker build --platform linux/amd64 -t ${IMAGE_NAME}:local .

4. Smoke-test the local image imports.
   docker run --rm ${IMAGE_NAME}:local python -c "import torch, transformers, datasets; print(torch.__version__)"

5. Tag and push the image to Nebius Registry.
   docker tag ${IMAGE_NAME}:local ${IMAGE_URI}
   docker push ${IMAGE_URI}

6. Create a Nebius Managed Kubernetes cluster and a two-node GPU node group.
   Final node group used here:
   ${NODE_GROUP_ID}

7. Install the NVIDIA GPU Operator so Kubernetes exposes nvidia.com/gpu.
   helm install gpu-operator oci://cr.${REGION}.nebius.cloud/marketplace/nebius/nvidia-gpu-operator/chart/gpu-operator \\
     --version v25.10.0 \\
     --set driver.version=580.95.05 \\
     -n nvidia-gpu-operator \\
     --create-namespace \\
     --wait

8. Export the node-group config deliverable.
   ${NEBIUS} mk8s node-group get --id ${NODE_GROUP_ID} --format json | jq '{metadata, spec}' > mk8s-ng-config.json

9. Verify SkyPilot can see the Kubernetes context.
   ${SKY} check kubernetes

10. Launch the distributed DDP training job.
   ${SKY} launch -c ${SKY_CLUSTER} train_job.yaml -y

11. If rerunning on an existing SkyPilot cluster, execute the job again.
   ${SKY} exec ${SKY_CLUSTER} train_job.yaml

12. Capture the successful job log. Set SKY_JOB_ID only if you need a specific historical job.
   ${SKY} logs ${SKY_CLUSTER} > training_log.txt

13. Remove terminal color codes from the log for a cleaner submission file.
   perl -0pi -e 's/\\x1b\\[[0-9;?]*[ -\\/]*[@-~]//g' training_log.txt

14. Package exactly the five required deliverables.
   zip -D ${SUBMISSION_ZIP} mk8s-ng-config.json Dockerfile train.py train_job.yaml training_log.txt

15. Verify the archive contents.
   unzip -l ${SUBMISSION_ZIP}

16. Shut down the SkyPilot runtime after logs are saved.
   ${SKY} down ${SKY_CLUSTER} -y

17. Delete paid cloud resources after the assignment is complete.
   CONFIRM_DELETE_CLOUD=1 ./run-full-project.sh cleanup-cloud

18. Verify that no project resources remain.
   ./run-full-project.sh verify-cloud-cleanup

Note: The Kubernetes GPU node group is separate from the SkyPilot runtime. Running cleanup-sky alone is not enough to stop all cloud charges.
STEPS
}

verify_local() {
  python -m py_compile train.py scripts/validate_project.py scripts/generate_run_summary.py
  python scripts/generate_run_summary.py
  python scripts/validate_project.py

  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git diff --exit-code RUN_SUMMARY.md >/dev/null || {
      echo "RUN_SUMMARY.md is stale; run './run-full-project.sh summarize-run' and commit the update."
      return 1
    }
  fi

  if [[ -f "$SUBMISSION_ZIP" ]]; then
    python scripts/validate_project.py --zip "$SUBMISSION_ZIP"
    zipinfo -1 "$SUBMISSION_ZIP"
  else
    echo "$SUBMISSION_ZIP is not present; run './run-full-project.sh package-submission' to recreate it."
  fi
}

docker_build() {
  docker build --platform linux/amd64 -t "${IMAGE_NAME}:local" .
}

docker_push() {
  docker tag "${IMAGE_NAME}:local" "$IMAGE_URI"
  docker push "$IMAGE_URI"
}

export_node_group_config() {
  "$NEBIUS" mk8s node-group get --id "$NODE_GROUP_ID" --format json \
    | jq '{metadata, spec}' > mk8s-ng-config.json
}

sky_check() {
  "$SKY" check kubernetes
}

sky_launch() {
  "$SKY" launch -c "$SKY_CLUSTER" train_job.yaml -y
}

capture_logs() {
  if [[ -n "$SKY_JOB_ID" ]]; then
    "$SKY" logs "$SKY_CLUSTER" "$SKY_JOB_ID" > training_log.txt
  else
    "$SKY" logs "$SKY_CLUSTER" > training_log.txt
  fi
  perl -0pi -e 's/\x1b\[[0-9;?]*[ -\/]*[@-~]//g' training_log.txt
}

summarize_run() {
  python scripts/generate_run_summary.py
}

package_submission() {
  rm -f "$SUBMISSION_ZIP"
  zip -D "$SUBMISSION_ZIP" mk8s-ng-config.json Dockerfile train.py train_job.yaml training_log.txt
  python scripts/validate_project.py --zip "$SUBMISSION_ZIP"
  zipinfo -1 "$SUBMISSION_ZIP"
}

cleanup_sky() {
  "$SKY" down "$SKY_CLUSTER" -y
}

require_cloud_delete_confirmation() {
  if [[ "${CONFIRM_DELETE_CLOUD:-}" != "1" ]]; then
    cat >&2 <<WARNING
Refusing to delete cloud resources without explicit confirmation.

This command deletes real Nebius resources:
  - SkyPilot runtime: ${SKY_CLUSTER}
  - Kubernetes cluster: ${CLUSTER_ID}
  - GPU node group under that cluster
  - Container registry: ${REGISTRY_RESOURCE_ID}
  - Registry artifacts/images

Run this exact command when you are ready:
  CONFIRM_DELETE_CLOUD=1 $0 cleanup-cloud
WARNING
    exit 2
  fi
}

resource_exists() {
  local resource_type="$1"
  local resource_id="$2"

  "$NEBIUS" "$resource_type" get --id "$resource_id" --format json >/dev/null 2>&1
}

delete_cluster_if_exists() {
  if resource_exists "mk8s cluster" "$CLUSTER_ID"; then
    "$NEBIUS" mk8s cluster delete --id "$CLUSTER_ID"
  else
    echo "Kubernetes cluster ${CLUSTER_ID} is already absent."
  fi
}

delete_registry_if_exists() {
  if ! resource_exists "registry" "$REGISTRY_RESOURCE_ID"; then
    echo "Container registry ${REGISTRY_RESOURCE_ID} is already absent."
    return
  fi

  local artifact_ids
  artifact_ids="$(
    "$NEBIUS" registry image list --parent-id "$REGISTRY_RESOURCE_ID" --format json \
      | jq -r '.items[]?.id'
  )"

  if [[ -n "$artifact_ids" ]]; then
    while IFS= read -r artifact_id; do
      [[ -z "$artifact_id" ]] && continue
      "$NEBIUS" registry image delete --id "$artifact_id" || true
    done <<< "$artifact_ids"
  fi

  artifact_ids="$(
    "$NEBIUS" registry image list --parent-id "$REGISTRY_RESOURCE_ID" --format json \
      | jq -r '.items[]?.id'
  )"

  if [[ -n "$artifact_ids" ]]; then
    while IFS= read -r artifact_id; do
      [[ -z "$artifact_id" ]] && continue
      "$NEBIUS" registry image delete --id "$artifact_id"
    done <<< "$artifact_ids"
  fi

  "$NEBIUS" registry delete --id "$REGISTRY_RESOURCE_ID"
}

cleanup_cloud() {
  require_cloud_delete_confirmation

  cleanup_sky || true
  delete_cluster_if_exists
  delete_registry_if_exists
  verify_cloud_cleanup
}

verify_empty_list() {
  local label="$1"
  shift

  local output
  output="$("$@" --format json)"

  if [[ "$output" == "{}" || "$output" == '{"items":[]}' ]]; then
    echo "OK: ${label}: none"
    return
  fi

  local count
  count="$(jq '(.items // []) | length' <<< "$output")"
  if [[ "$count" == "0" ]]; then
    echo "OK: ${label}: none"
  else
    echo "WARNING: ${label}: ${count} resource(s) still present"
    jq . <<< "$output"
    return 1
  fi
}

verify_cloud_cleanup() {
  "$SKY" status || true
  verify_empty_list "Kubernetes clusters" "$NEBIUS" mk8s cluster list --parent-id "$PROJECT_ID"
  verify_empty_list "Container registries" "$NEBIUS" registry list --parent-id "$PROJECT_ID"
  verify_empty_list "Compute instances" "$NEBIUS" compute instance list --parent-id "$PROJECT_ID"
  verify_empty_list "Compute disks" "$NEBIUS" compute disk list --parent-id "$PROJECT_ID"
}

usage() {
  cat <<USAGE
Usage: $0 <command>

Commands:
  print-steps          Print the full project procedure.
  verify-local         Validate local files and submission evidence.
  docker-build         Build the training image locally.
  docker-push          Tag and push the image to Nebius Registry.
  export-node-group    Recreate mk8s-ng-config.json from Nebius.
  sky-check            Verify SkyPilot Kubernetes access.
  sky-launch           Launch the two-node SkyPilot DDP job.
  capture-logs         Fetch SkyPilot logs into training_log.txt.
  summarize-run        Generate RUN_SUMMARY.md from training_log.txt.
  package              Build the five-file submission zip.
  package-submission   Same as package; explicit assignment archive command.
  cleanup-sky          Shut down the SkyPilot runtime.
  cleanup-cloud        Delete SkyPilot, Kubernetes, registry images, and registry.
                       Requires CONFIRM_DELETE_CLOUD=1.
  verify-cloud-cleanup Verify no clusters, registries, instances, or disks remain.
  all                  Run the cloud pipeline from Docker build through packaging.

The all command can create paid cloud/GPU usage. Check the variables at the top
of this file before running it.
USAGE
}

main() {
  local command="${1:-print-steps}"

  case "$command" in
    print-steps) print_steps ;;
    verify-local) verify_local ;;
    docker-build) docker_build ;;
    docker-push) docker_push ;;
    export-node-group) export_node_group_config ;;
    sky-check) sky_check ;;
    sky-launch) sky_launch ;;
    capture-logs) capture_logs ;;
    summarize-run) summarize_run ;;
    package|package-submission) package_submission ;;
    cleanup-sky) cleanup_sky ;;
    cleanup-cloud) cleanup_cloud ;;
    verify-cloud-cleanup) verify_cloud_cleanup ;;
    all)
      docker_build
      docker_push
      export_node_group_config
      sky_check
      sky_launch
      capture_logs
      summarize_run
      package_submission
      cleanup_sky
      ;;
    help|-h|--help) usage ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
