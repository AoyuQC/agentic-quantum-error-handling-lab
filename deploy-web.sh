#!/usr/bin/env bash
# Deploy the Agentic QEM Lab WEB UI to AWS as a public, cloud-native app:
#   Browser -> CloudFront (HTTPS) -> ALB -> ECS Fargate (aqem-web).
#
# This is independent of the AgentCore deployment (deploy/deploy.sh). CDK builds
# the Dockerfile.web image, pushes it to ECR, and stands up the whole stack.
#
# Prereqs: AWS creds; Docker running; Node/npm; Bedrock model access for Claude
#          (only needed if users toggle "Use VLM" on).
#
# Usage:
#   ./deploy-web.sh            # deploy (default) — prints the public CloudFront URL
#   ./deploy-web.sh deploy
#   ./deploy-web.sh diff
#   ./deploy-web.sh destroy    # tear the web stack down
set -euo pipefail

REGION="${REGION:-us-east-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDK_DIR="${SCRIPT_DIR}/cdk"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
export CDK_DEFAULT_ACCOUNT="$ACCOUNT"
export CDK_DEFAULT_REGION="$REGION"

cmd="${1:-deploy}"
cd "$CDK_DIR"

if [ ! -d node_modules ]; then
  echo ">> installing CDK dependencies"
  npm install
fi

case "$cmd" in
  deploy)
    echo ">> bootstrapping CDK (idempotent) in aws://${ACCOUNT}/${REGION}"
    npx cdk bootstrap "aws://${ACCOUNT}/${REGION}"
    echo ">> deploying AqemWebStack (builds Dockerfile.web -> ECR -> Fargate)"
    # Non-interactive: the script runs unattended (CI / background). The stack's
    # IAM + SG changes are reviewed in this repo; `cdk diff` shows them anytime.
    npx cdk deploy --require-approval never --outputs-file cdk-outputs.json
    echo
    echo ">> done. Public URL:"
    node -e "const o=require('./cdk-outputs.json');console.log('   '+o.AqemWebStack.CloudFrontURL)" 2>/dev/null \
      || echo "   (see the CloudFrontURL output above)"
    ;;
  diff)
    npx cdk diff
    ;;
  destroy)
    echo ">> destroying AqemWebStack"
    npx cdk destroy --force
    ;;
  *)
    echo "usage: $0 {deploy|diff|destroy}   env: REGION=$REGION"
    ;;
esac
