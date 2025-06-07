"""Simple stub for LLM client creation."""
from __future__ import annotations

from typing import Any, Optional, List, Dict


class DummyLLMClient:
    async def generate(self, messages: List[Dict[str, str]], **params: Any) -> str:
        # Echo back the last user message for testing
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""


def create_llm_client(provider: str, api_key: Optional[str] = None) -> DummyLLMClient:
    """Return a dummy LLM client for tests."""
    return DummyLLMClient()
