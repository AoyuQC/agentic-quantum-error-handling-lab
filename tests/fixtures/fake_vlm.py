"""A deterministic VLMProvider returning canned responses.

Lets the AI-dependent paths (empirical_probe classification, validate verdict)
run offline and reproducibly in tests. Implements the same ``analyze_images``
interface as the real providers in ``aqem.vlm.providers``.
"""

from __future__ import annotations

import json
from typing import List

from aqem.vlm.providers import VLMProvider


class FakeVLM(VLMProvider):
    """Returns a fixed string (optionally JSON) regardless of the input images."""

    def __init__(self, response: str | dict):
        self.response = json.dumps(response) if isinstance(response, dict) else response
        self.calls: list[dict] = []

    async def analyze_images(
        self, prompt: str, images_base64: List[str], image_format: str = "png"
    ) -> str:
        self.calls.append({"prompt": prompt, "n_images": len(images_base64)})
        return self.response
