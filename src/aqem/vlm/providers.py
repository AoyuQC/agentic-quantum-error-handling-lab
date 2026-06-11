# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from NVIDIA Quantum-Calibration-Agent-Blueprint (vlm/providers.py).
# See the repository NOTICE file for attribution.
#
# Adaptation: BedrockClaudeProvider (the DEFAULT) and SageMakerProvider are
# added here in Phase L3, alongside the upstream NVIDIA / Anthropic / custom
# providers. The factory dispatches "bedrock" by default.

"""VLM provider abstraction.

Defines a single ``VLMProvider`` interface so the agent's vision calls are
provider-pluggable. The default provider will be Bedrock Claude (managed model);
the small NVIDIA Ising-Calibration VLM on a SageMaker endpoint is a later,
config-only swap.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


class VLMProvider(ABC):
    """Abstract base class for VLM providers."""

    @abstractmethod
    async def analyze_images(
        self,
        prompt: str,
        images_base64: List[str],
        image_format: str = "png",
    ) -> str:
        """Analyze images with the VLM.

        Args:
            prompt: Analysis instructions
            images_base64: List of base64-encoded images
            image_format: Image format (png, jpeg)

        Returns:
            Analysis text from the VLM
        """
        pass


class NVIDIAProvider(VLMProvider):
    """VLM provider using official langchain-nvidia-ai-endpoints."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 8192,
        enable_thinking: Optional[bool] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env or "", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

    async def analyze_images(
        self,
        prompt: str,
        images_base64: List[str],
        image_format: str = "png",
    ) -> str:
        """Analyze images using NVIDIA endpoint."""
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            from langchain_core.messages import HumanMessage
        except ImportError as e:
            raise RuntimeError(
                "langchain-nvidia-ai-endpoints is required. Install with: pip install langchain-nvidia-ai-endpoints"
            ) from e

        content = [{"type": "text", "text": prompt}]
        for img_b64 in images_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{img_b64}"
                    },
                }
            )

        try:
            kwargs = dict(
                model=self.model,
                api_key=self.api_key if self.api_key else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if self.enable_thinking is not None:
                kwargs["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
            chat = ChatNVIDIA(**kwargs)
            message = HumanMessage(content=content)
            response = await chat.ainvoke([message])
            return response.content
        except Exception as e:
            logger.error(f"NVIDIA VLM call failed: {e}")
            raise RuntimeError(f"VLM analysis failed: {e}") from e


class AnthropicProvider(VLMProvider):
    """VLM provider using Anthropic Claude models (direct API, for offline dev)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env or "", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def analyze_images(
        self,
        prompt: str,
        images_base64: List[str],
        image_format: str = "png",
    ) -> str:
        """Analyze images using Anthropic Claude."""
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage
        except ImportError as e:
            raise RuntimeError(
                "langchain-anthropic is required. Install with: pip install langchain-anthropic"
            ) from e

        content = []
        for img_b64 in images_base64:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{image_format}",
                        "data": img_b64,
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        try:
            chat = ChatAnthropic(
                model=self.model,
                api_key=self.api_key if self.api_key else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            message = HumanMessage(content=content)
            response = await chat.ainvoke([message])
            return response.content
        except Exception as e:
            logger.error(f"Anthropic VLM call failed: {e}")
            raise RuntimeError(f"VLM analysis failed: {e}") from e


class CustomEndpointProvider(VLMProvider):
    """VLM provider for custom OpenAI-compatible endpoints."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 8192,
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env or "", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def analyze_images(
        self,
        prompt: str,
        images_base64: List[str],
        image_format: str = "png",
    ) -> str:
        """Analyze images using custom endpoint."""
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError(
                "httpx is required for custom endpoints. Install with: pip install httpx"
            ) from e

        content = [{"type": "text", "text": prompt}]
        for img_b64 in images_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_format};base64,{img_b64}"},
                }
            )

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": content}],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Custom endpoint call failed: {e}")
            raise RuntimeError(f"VLM analysis failed: {e}") from e


class BedrockClaudeProvider(VLMProvider):
    """Default VLM provider — managed Claude on Amazon Bedrock.

    Uses ``langchain_aws.ChatBedrockConverse`` with the standard AWS credential
    chain (profile / env / instance role) — no static keys, so it is
    AgentCore-Identity-ready. Multimodal content is sent as Bedrock Converse
    image blocks (``{"type": "image", "base64": ..., "mimeType": ...}``).
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 4096,
    ):
        self.model_id = model_id
        self.region = region
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _client(self):
        from langchain_aws import ChatBedrockConverse

        kwargs = dict(model=self.model_id, temperature=self.temperature, max_tokens=self.max_tokens)
        if self.region:
            kwargs["region_name"] = self.region
        return ChatBedrockConverse(**kwargs)

    async def analyze_images(
        self,
        prompt: str,
        images_base64: List[str],
        image_format: str = "png",
    ) -> str:
        try:
            from langchain_core.messages import HumanMessage
        except ImportError as e:  # pragma: no cover - langchain-core is a hard dep
            raise RuntimeError("langchain-core is required for the Bedrock provider") from e

        # langchain-core v1 canonical data-content-block shape. (Note: the
        # short {"type":"image","base64":...,"mimeType":...} form is mis-handled
        # by current langchain-aws; this source_type/data/mime_type form is the
        # one that survives the full Converse conversion — verified live.)
        content: list[dict] = []
        for img_b64 in images_base64:
            content.append(
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": img_b64,
                    "mime_type": f"image/{image_format}",
                }
            )
        content.append({"type": "text", "text": prompt})

        try:
            chat = self._client()
            response = await chat.ainvoke([HumanMessage(content=content)])
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception as e:
            logger.error(f"Bedrock VLM call failed: {e}")
            raise RuntimeError(f"VLM analysis failed: {e}") from e


class SageMakerProvider(VLMProvider):
    """VLM provider for a SageMaker real-time endpoint (the NVIDIA
    Ising-Calibration VLM) — the later, config-only swap from Bedrock Claude.

    Invokes the endpoint via boto3 ``sagemaker-runtime``. The request/response
    payload mirrors an OpenAI-style chat-completions schema with base64 image
    URLs; adjust ``payload`` to match the deployed container's contract.
    """

    def __init__(
        self,
        endpoint_name: str,
        region: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 8192,
        model: str = "ising-calibration",
    ):
        self.endpoint_name = endpoint_name
        self.region = region
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model

    async def analyze_images(
        self,
        prompt: str,
        images_base64: List[str],
        image_format: str = "png",
    ) -> str:
        import asyncio
        import json

        try:
            import boto3
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("boto3 is required for the SageMaker provider") from e

        content = [{"type": "text", "text": prompt}]
        for img_b64 in images_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_format};base64,{img_b64}"},
                }
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        def _invoke() -> str:
            client = boto3.client("sagemaker-runtime", region_name=self.region)
            resp = client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            body = json.loads(resp["Body"].read())
            # OpenAI-style response; fall back to the raw body if shaped otherwise.
            try:
                return body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                return body if isinstance(body, str) else json.dumps(body)

        try:
            # boto3 is synchronous; run it off the event loop.
            return await asyncio.to_thread(_invoke)
        except Exception as e:
            logger.error(f"SageMaker VLM call failed: {e}")
            raise RuntimeError(f"VLM analysis failed: {e}") from e


def get_vlm_client(config: dict) -> VLMProvider:
    """Create VLM client from configuration.

    Args:
        config: VLM configuration dict with keys:
            - provider: "bedrock" (default), "sagemaker", "anthropic", "nvidia", or "custom"
            - model / model_id: Model name/string
            - api_key or api_key_env: API key configuration
            - endpoint: (for custom) API endpoint URL
            - temperature: Sampling temperature
            - max_tokens: Max response tokens

    Returns:
        VLMProvider instance
    """
    provider = config.get("provider", "bedrock")

    if provider == "bedrock":
        return BedrockClaudeProvider(
            model_id=config.get("model_id", config.get("model", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")),
            region=config.get("region"),
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", 4096),
        )
    elif provider == "sagemaker":
        return SageMakerProvider(
            endpoint_name=config["endpoint_name"],
            region=config.get("region"),
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", 8192),
            model=config.get("model", "ising-calibration"),
        )
    elif provider in ("nvidia", "litellm"):  # litellm for backwards compat
        return NVIDIAProvider(
            model=config.get("model", "nvidia/ising-calibration-1-35b-a3b"),
            api_key=config.get("api_key"),
            api_key_env=config.get("api_key_env"),
            temperature=config.get("temperature", 0.2),
            max_tokens=config.get("max_tokens", 32768),
            enable_thinking=config.get("enable_thinking"),
        )
    elif provider == "anthropic":
        return AnthropicProvider(
            model=config.get("model", "claude-sonnet-4-6"),
            api_key=config.get("api_key"),
            api_key_env=config.get("api_key_env"),
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", 4096),
        )
    elif provider == "custom":
        return CustomEndpointProvider(
            endpoint=config["endpoint"],
            model=config.get("model", ""),
            api_key=config.get("api_key"),
            api_key_env=config.get("api_key_env"),
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", 4096),
        )
    else:
        raise ValueError(
            f"Unknown VLM provider: {provider}. "
            "Choose from: bedrock, sagemaker, anthropic, nvidia, custom."
        )
