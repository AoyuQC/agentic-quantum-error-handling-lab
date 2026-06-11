# Amazon Bedrock AgentCore Runtime container for the Agentic QEM Lab.
# AgentCore Runtime requires linux/arm64. Build with:
#   docker buildx build --platform linux/arm64 -t aqem-agent .
# (The `agentcore deploy` CodeBuild path produces this image for you.)

FROM --platform=linux/arm64 public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

# System deps: kaleido 0.2.1 bundles chromium but needs a few shared libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libnss3 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Install the package first (better layer caching).
COPY pyproject.toml requirements.txt README.md NOTICE ./
COPY src ./src
RUN pip install --no-cache-dir -e .

ENV AWS_REGION=us-east-1 \
    AQEM_DEVICE=qd_readout_2 \
    PYTHONUNBUFFERED=1

# AgentCore Runtime invokes the BedrockAgentCoreApp server on port 8080.
EXPOSE 8080
CMD ["python", "-m", "aqem.cloud.runtime"]
