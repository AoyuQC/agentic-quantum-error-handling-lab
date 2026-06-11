#!/usr/bin/env bash
# Provision + deploy the Agentic QEM Lab to Amazon Bedrock AgentCore Runtime.
#
# Prereqs: AWS creds with permission to create S3/IAM + use Bedrock & AgentCore;
#          `pip install bedrock-agentcore bedrock-agentcore-starter-toolkit`;
#          Bedrock model access enabled for Claude Sonnet 4.5 in $REGION.
#
# Usage:
#   REGION=us-east-1 ./deploy/deploy.sh provision   # create S3 artifact bucket
#   ./deploy/deploy.sh configure                     # agentcore configure
#   ./deploy/deploy.sh deploy                        # build (CodeBuild) + deploy
#   ./deploy/deploy.sh invoke                        # smoke-invoke the runtime
#   ./deploy/deploy.sh destroy                       # tear everything down
set -euo pipefail

REGION="${REGION:-us-east-1}"
AGENT_NAME="${AGENT_NAME:-aqem}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${BUCKET:-aqem-artifacts-${ACCOUNT}-${REGION}}"
MODEL_ID="${MODEL_ID:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
export AGENTCORE_SUPPRESS_RECOMMENDATION=1

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

  destroy)
    echo ">> destroying AgentCore resources for ${AGENT_NAME}"
    agentcore destroy --agent "$AGENT_NAME" || true
    echo ">> emptying + deleting s3://${BUCKET}"
    aws s3 rm "s3://${BUCKET}" --recursive || true
    aws s3api delete-bucket --bucket "$BUCKET" --region "$REGION" || true
    ;;

  *)
    echo "usage: $0 {provision|configure|deploy|invoke|status|destroy}"
    echo "env: REGION=$REGION AGENT_NAME=$AGENT_NAME BUCKET=$BUCKET MODEL_ID=$MODEL_ID"
    ;;
esac
