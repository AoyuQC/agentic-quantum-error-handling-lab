"""AgentCore Runtime entrypoint module (target for `agentcore configure`).

Re-exports the BedrockAgentCoreApp ``app`` built in ``aqem.cloud.runtime`` so the
AgentCore toolkit can discover it. Keeping it at the repo root is the convention
the CLI auto-detects.

    agentcore configure --entrypoint agent.py --name aqem
    agentcore deploy
"""

from aqem.cloud.runtime import app  # noqa: F401

if __name__ == "__main__":
    if app is not None:
        app.run()
    else:  # pragma: no cover
        raise SystemExit("bedrock-agentcore SDK not installed; pip install bedrock-agentcore")
