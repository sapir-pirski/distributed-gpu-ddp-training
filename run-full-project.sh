#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-eu-north1}"
REGISTRY_ID="${REGISTRY_ID:-e00avpz7r2gn4zffdk}"
IMAGE_NAME="${IMAGE_NAME:-nebius-trainer}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
IMAGE_URI="${IMAGE_URI:-cr.${REGION}.nebius.cloud/${REGISTRY_ID}/${IMAGE_NAME}:${IMAGE_TAG}}"

CLUSTER_CONTEXT="${CLUSTER_CONTEXT:-nebius-ddp-mk8s}"
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

Note: The Kubernetes GPU node group is separate from the SkyPilot runtime and may still need to be deleted or scaled down in Nebius to stop GPU charges.
STEPS
}

verify_local() {
  python -m py_compile train.py
  jq . mk8s-ng-config.json >/dev/null
  rg -n "World size: 2|NCCL|Training complete|status: SUCCEEDED" training_log.txt
  if [[ -f "$SUBMISSION_ZIP" ]]; then
    zipinfo -1 "$SUBMISSION_ZIP"
  else
    echo "$SUBMISSION_ZIP is not present; run './run-full-project.sh package' to recreate it."
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

package_submission() {
  zip -D "$SUBMISSION_ZIP" mk8s-ng-config.json Dockerfile train.py train_job.yaml training_log.txt
  unzip -l "$SUBMISSION_ZIP"
}

cleanup_sky() {
  "$SKY" down "$SKY_CLUSTER" -y
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
  package              Build the five-file submission zip.
  cleanup-sky          Shut down the SkyPilot runtime.
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
    package) package_submission ;;
    cleanup-sky) cleanup_sky ;;
    all)
      docker_build
      docker_push
      export_node_group_config
      sky_check
      sky_launch
      capture_logs
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
