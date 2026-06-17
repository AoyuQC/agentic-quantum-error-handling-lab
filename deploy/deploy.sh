#!/usr/bin/env bash
# Provision + deploy the Agentic QEM Lab to Amazon Bedrock AgentCore Runtime.
#
# Prereqs: AWS creds with permission to create S3/IAM + use Bedrock & AgentCore;
#          `pip install bedrock-agentcore bedrock-agentcore-starter-toolkit`;
#          Bedrock model access enabled for Claude Opus 4.8 in $REGION.
#
# Usage:
#   REGION=us-east-1 ./deploy/deploy.sh provision   # create S3 artifact bucket
#   ./deploy/deploy.sh configure                     # agentcore configure (aqem runtime)
#   ./deploy/deploy.sh deploy                        # build (CodeBuild) + deploy
#   ./deploy/deploy.sh invoke                        # smoke-invoke the runtime
#   ./deploy/deploy.sh destroy                       # tear everything down
#
# Gateway (MCP tool server) — optional, deploys the tools as a separate
# MCP-protocol Runtime and points the aqem runtime at it:
#   ./deploy/deploy.sh configure-mcp                 # agentcore configure --protocol MCP
#   ./deploy/deploy.sh deploy-mcp                    # build + deploy the MCP server
#   ./deploy/deploy.sh status-mcp                    # MCP runtime status + ARN
#   MCP_ENDPOINT=<url> ./deploy/deploy.sh deploy-gw  # redeploy aqem with mcp transport
set -euo pipefail

REGION="${REGION:-us-east-1}"
AGENT_NAME="${AGENT_NAME:-aqem}"
TOOLS_AGENT="${TOOLS_AGENT:-aqem_tools}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${BUCKET:-aqem-artifacts-${ACCOUNT}-${REGION}}"
MODEL_ID="${MODEL_ID:-us.anthropic.claude-opus-4-8}"
export AGENTCORE_SUPPRESS_RECOMMENDATION=1

# The starter toolkit always builds from the root file named `Dockerfile`. The
# MCP tool server needs a different image (port 8000, runs aqem.cloud.mcp_server),
# so for the MCP agent we temporarily swap Dockerfile.mcp into place and restore
# the original on exit.
with_mcp_dockerfile() {
  [ -f Dockerfile.mcp ] || { echo "Dockerfile.mcp not found"; exit 1; }
  local backup="Dockerfile.runtime.bak"
  [ -f Dockerfile ] && cp Dockerfile "$backup"
  cp Dockerfile.mcp Dockerfile
  # shellcheck disable=SC2064
  trap "[ -f '$backup' ] && mv '$backup' Dockerfile" EXIT
  "$@"
}

cmd="${1:-help}"
case "$cmd" in
  provision)
    echo ">> creating artifact bucket s3://${BUCKET} in ${REGION}"
    if [ "$REGION" = "us-east-1" ]; then
      aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" 2>/dev/null || true
    else
      aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION" 2>/dev/null || true
    fi
    aws s3api put-public-access-block --bucket "$BUCKET" \
      --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    echo ">> bucket ready: s3://${BUCKET}"
    echo ">> NOTE: attach deploy/execution-role-policy.json to the runtime execution role"
    echo "         (substitute \${BUCKET}=${BUCKET}, \${REGION}=${REGION}, \${ACCOUNT}=${ACCOUNT})"
    ;;

  configure)
    echo ">> configuring AgentCore (entrypoint agent.py, name ${AGENT_NAME})"
    agentcore configure --entrypoint agent.py --name "$AGENT_NAME" --region "$REGION"
    echo ">> review .bedrock_agentcore.yaml before deploying"
    ;;

  deploy)
    echo ">> deploying ${AGENT_NAME} to AgentCore Runtime in ${REGION} (CodeBuild ARM64)"
    agentcore deploy --agent "$AGENT_NAME" \
      --env "AWS_REGION=${REGION}" \
      --env "AQEM_VLM_MODEL_ID=${MODEL_ID}" \
      --env "AQEM_ARTIFACTS=s3://${BUCKET}/aqem" \
      --env "AQEM_DEVICE=${AGENT_DEVICE:-qd_readout_2}"
    ;;

  invoke)
    echo ">> invoking ${AGENT_NAME}"
    agentcore invoke --agent "$AGENT_NAME" \
      '{"qubits": 2, "target_accuracy": 0.06, "device": "qd_readout_2", "seed": 7}'
    ;;

  status)
    agentcore status --agent "$AGENT_NAME"
    ;;

  # --- Gateway: MCP tool server (separate MCP-protocol Runtime) ------------
  configure-mcp)
    echo ">> configuring AgentCore MCP server (entrypoint agent_mcp.py, name ${TOOLS_AGENT})"
    with_mcp_dockerfile agentcore configure --entrypoint agent_mcp.py \
      --name "$TOOLS_AGENT" --region "$REGION" --protocol MCP
    echo ">> review .bedrock_agentcore.yaml before deploying"
    ;;

  deploy-mcp)
    echo ">> deploying ${TOOLS_AGENT} (MCP tool server) to AgentCore Runtime in ${REGION}"
    with_mcp_dockerfile agentcore deploy --agent "$TOOLS_AGENT" \
      --env "AWS_REGION=${REGION}" \
      --env "AQEM_VLM_MODEL_ID=${MODEL_ID}" \
      --env "AQEM_DEVICE=${AGENT_DEVICE:-qd_readout_2}"
    ;;

  status-mcp)
    agentcore status --agent "$TOOLS_AGENT"
    ;;

  # Redeploy the aqem runtime wired to route tool calls through the MCP server.
  # Pass the MCP endpoint URL via MCP_ENDPOINT (see `status-mcp` for the ARN).
  deploy-gw)
    : "${MCP_ENDPOINT:?set MCP_ENDPOINT=<mcp server url> (see status-mcp)}"
    echo ">> deploying ${AGENT_NAME} with mcp tool transport -> ${MCP_ENDPOINT}"
    agentcore deploy --agent "$AGENT_NAME" \
      --env "AWS_REGION=${REGION}" \
      --env "AQEM_VLM_MODEL_ID=${MODEL_ID}" \
      --env "AQEM_ARTIFACTS=s3://${BUCKET}/aqem" \
      --env "AQEM_DEVICE=${AGENT_DEVICE:-qd_readout_2}" \
      --env "AQEM_TOOL_TRANSPORT=mcp" \
      --env "AQEM_MCP_ENDPOINT=${MCP_ENDPOINT}"
    ;;

  destroy)
    echo ">> destroying AgentCore resources for ${AGENT_NAME} and ${TOOLS_AGENT}"
    # --force skips the interactive y/N prompt; --delete-ecr-repo removes the
    # per-agent ECR repository too (otherwise it lingers and incurs cost).
    agentcore destroy --agent "$AGENT_NAME" --force --delete-ecr-repo || true
    agentcore destroy --agent "$TOOLS_AGENT" --force --delete-ecr-repo || true
    echo ">> emptying + deleting s3://${BUCKET}"
    aws s3 rm "s3://${BUCKET}" --recursive || true
    aws s3api delete-bucket --bucket "$BUCKET" --region "$REGION" || true
    ;;

  *)
    echo "usage: $0 {provision|configure|deploy|invoke|status|destroy}"
    echo "       $0 {configure-mcp|deploy-mcp|status-mcp|deploy-gw}"
    echo "env: REGION=$REGION AGENT_NAME=$AGENT_NAME TOOLS_AGENT=$TOOLS_AGENT BUCKET=$BUCKET MODEL_ID=$MODEL_ID"
    ;;
esac
