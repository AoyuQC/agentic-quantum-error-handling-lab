# AQEM Lab — web UI deployment (AWS CDK)

Deploys the Agentic QEM Lab **web UI + API** as a public, cloud-native app:

```
Browser ──HTTPS──> CloudFront ──HTTP (+secret header)──> ALB ──> ECS Fargate
                   (public URL)    (CloudFront-only)            aqem-web :8000
                                                                (UI + /api/*)
```

A single Fargate task runs `aqem-web` (FastAPI/Uvicorn), serving the built React
SPA (`ui/dist`) and the `/api/*` endpoints. The adaptive QEM loop runs in-process
on a local Braket noise simulator — **no AWS calls for the core**. Bedrock is only
hit when a user toggles **Use VLM** on; the Fargate task role allows that.

This stack is **independent of the AgentCore deployment** (`deploy/deploy.sh`).
You can run one, the other, or neither.

## Deploy

From the repo root (needs AWS creds + a running Docker daemon):

```bash
./deploy-web.sh            # bootstrap + deploy, prints the public CloudFront URL
```

CDK builds `Dockerfile.web` (multi-stage: Node builds `ui/dist`, Python serves it),
pushes the image to ECR, and stands up VPC + ECS Fargate + ALB + CloudFront.
First deploy takes ~10–15 min (image build + CloudFront propagation).

Open the `CloudFrontURL` output in a browser.

## Destroy

```bash
./deploy-web.sh destroy
```

Removes the stack (NAT gateway, ALB, Fargate, CloudFront) and the artifact bucket.

## Notes

- **SSE:** `/api/run` streams progress over Server-Sent Events. The `/api/*`
  CloudFront behavior is `CACHING_DISABLED` + `ALL_VIEWER`, and the ALB idle
  timeout is 300s, so live progress isn't buffered or dropped.
- **Security:** the ALB only accepts requests carrying a secret header that
  CloudFront injects; direct `http://<ALB-DNS>/` returns **403**. The ALB SG
  allows inbound only from the CloudFront managed prefix list.
- **Networking:** to avoid consuming an Elastic IP (and a NAT gateway's cost),
  the Fargate task runs in a **public subnet with a public IP** for egress
  (ECR/Bedrock). Its security group permits inbound **only from the ALB**, so
  the task is never directly reachable despite having a public IP.
- **VLM prerequisite:** enable Bedrock model access for
  `us.anthropic.claude-opus-4-8` in the region, or leave the VLM toggle off.
- **Sizing:** task is 1 vCPU / 2 GB, autoscaling 1→3 on 70% CPU. Bump
  `cpu`/`memoryLimitMiB` in `lib/aqem-web-stack.ts` if runs feel slow.
