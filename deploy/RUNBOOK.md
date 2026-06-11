# Deployment Runbook — AQEM on Amazon Bedrock AgentCore

Deploys the adaptive QEM loop as a **Bedrock AgentCore Runtime** agent. The VLM
is **managed Claude on Bedrock** (no SageMaker). Large artifacts go to **S3**;
optional **Bedrock Guardrails** screen free-text I/O alongside the deterministic
`policy/` layer.

> The loop runs entirely in-process inside the Runtime container — the tools in
> `aqem/tools/` are the seams a Gateway MCP server would later route to. This
> runbook covers the in-process Runtime deployment.

## Prerequisites

- AWS credentials with permission for Bedrock, S3, IAM, and AgentCore.
- Region **us-east-1** (AgentCore Runtime GA; Claude Sonnet 4.5 available). Verified.
- Bedrock model access enabled for `anthropic.claude-sonnet-4-5` in the console.
- Tooling: `pip install -e ".[cloud]"` (installs `bedrock-agentcore` +
  `bedrock-agentcore-starter-toolkit`).
- Docker/Finch only if you use `--local`/`--local-build`; the default cloud
  deploy builds an ARM64 image via CodeBuild (no local Docker needed).

## One-time setup

```bash
export REGION=us-east-1
export AGENTCORE_SUPPRESS_RECOMMENDATION=1

# 1. Artifact bucket (idempotent).
./deploy/deploy.sh provision
#    -> s3://aqem-artifacts-<account>-us-east-1

# 2. Create the runtime execution role and attach the least-privilege policy.
#    Substitute ${BUCKET}/${REGION}/${ACCOUNT} in deploy/execution-role-policy.json.
#    Trust policy principal: bedrock-agentcore.amazonaws.com
```

## Configure + deploy

```bash
# 3. Configure the agent (entrypoint = agent.py, which exposes the BedrockAgentCoreApp).
./deploy/deploy.sh configure       # wraps: agentcore configure -e agent.py -n aqem
#    Review the generated .bedrock_agentcore.yaml (execution role, ECR repo, env).

# 4. Deploy (CodeBuild builds the ARM64 image, pushes to ECR, deploys to Runtime).
./deploy/deploy.sh deploy
#    Sets env: AWS_REGION, AQEM_VLM_MODEL_ID, AQEM_ARTIFACTS=s3://.../aqem, AQEM_DEVICE
```

## Invoke

```bash
./deploy/deploy.sh invoke
# or directly:
agentcore invoke --agent aqem \
  '{"qubits": 2, "target_accuracy": 0.06, "device": "qd_readout_2", "seed": 7}'
```

Response (abridged):

```json
{
  "status": "stopped",
  "estimate": {"value": 1.81, "techniques": ["REM"], "shots_used": 40000},
  "shots_used": 40000,
  "comparison": {"efficiency_gain_demonstrated": true, "shot_ratio": 2.1},
  "artifacts": {
    "audit": "s3://aqem-artifacts-<acct>-us-east-1/aqem/<run>/audit.json",
    "comparison": "s3://.../comparison.json",
    "accuracy_vs_shots": "s3://.../accuracy_vs_shots.json"
  }
}
```

### Invocation payload keys

| key | default | meaning |
|---|---|---|
| `qubits` | 2 | circuit size |
| `target_accuracy` | 0.06 | stop when |estimate − ideal| ≤ this |
| `device` | `qd_readout_2` | noise model (`qd_readout`, `qd_depol`, `qd_total`, …) |
| `budget_shots` | 2_000_000 | Policy hard shot ceiling |
| `use_vlm` | true | use the Bedrock Claude VLM (else rules-only) |
| `compare_baseline` | true | also run the static full-stack baseline |
| `seed` | — | RNG seed for the twirling (shot sampling is not seedable) |
| `guardrail_id` | — | optional Bedrock guardrail for text I/O |

## Observability

- AgentCore emits traces/logs to CloudWatch (log group created via the policy).
- Each response carries the full Policy **audit trail** (also persisted to S3),
  so every action is shown to be controlled-set + budget-gated.

## Local development

```bash
agentcore dev                       # local hot-reload server on :8080
# or, no SDK server, just the handler:
python -c "from aqem.cloud.runtime import invoke; import json; \
  print(json.dumps(invoke({'device':'qd_readout_2','seed':7}), indent=2, default=str))"
```

## Teardown

```bash
./deploy/deploy.sh destroy           # agentcore destroy + empty/delete the S3 bucket
```

## Cost notes

- AgentCore Runtime is serverless — pay per invocation/duration, no idle cost.
- Bedrock Claude is per-token; each run makes a small number of VLM calls.
- S3 artifacts are tiny JSON. Tear down with `destroy` after a demo.
