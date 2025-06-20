"""
LLM client implementations for various providers.
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
):
    """Decorator for retrying functions with exponential backoff."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = min(delay, max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s..."
                        )
                        await asyncio.sleep(sleep_time)
                        delay *= exponential_base
                    else:
                        logger.error(f"All {max_retries} attempts failed.")

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = min(delay, max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s..."
                        )
                        time.sleep(sleep_time)
                        delay *= exponential_base
                    else:
                        logger.error(f"All {max_retries} attempts failed.")

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def validate_api_key(self) -> bool:
        """Validate that the API key is set and valid."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"))
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Run: pip install openai"
                )
        return self._client

    @retry_with_exponential_backoff()
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> str:
        """Generate response using OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def generate_structured(
        self,
        messages: List[Dict[str, str]],
        model: str,
        schema: Dict[str, Any],
        temperature: float = 0.3,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate structured output using function calling."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                **kwargs,
            )

            import json

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI structured generation error: {e}")
            raise

    def validate_api_key(self) -> bool:
        """Validate OpenAI API key."""
        return bool(self.api_key and self.api_key.startswith("sk-"))


class AnthropicClient(BaseLLMClient):
    """Anthropic (Claude) API client."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"))
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    @retry_with_exponential_backoff()
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> str:
        """Generate response using Anthropic API."""
        try:
            system_message = None
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

            response = await self.client.messages.create(
                model=model,
                messages=user_messages,
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def validate_api_key(self) -> bool:
        """Validate Anthropic API key."""
        return bool(self.api_key)


class GoogleAIClient(BaseLLMClient):
    """Google AI (Gemini) API client."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("GOOGLE_API_KEY"))
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Google AI client."""
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError(
                    "Google AI package not installed. Run: pip install google-generativeai"
                )
        return self._client

    @retry_with_exponential_backoff()
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-1.5-pro",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> str:
        """Generate response using Google AI API."""
        try:
            genai_model = self.client.GenerativeModel(model)

            chat_history = []
            last_user_message = ""

            for msg in messages:
                if msg["role"] == "system":
                    continue
                elif msg["role"] == "user":
                    last_user_message = msg["content"]
                else:
                    chat_history.append(
                        {
                            "role": "user" if msg["role"] == "user" else "model",
                            "parts": [msg["content"]],
                        }
                    )

            system_content = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), ""
            )
            if system_content:
                last_user_message = f"{system_content}\n\n{last_user_message}"

            chat = genai_model.start_chat(history=chat_history)

            response = await asyncio.to_thread(
                chat.send_message,
                last_user_message,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    **kwargs,
                },
            )

            return response.text
        except Exception as e:
            logger.error(f"Google AI API error: {e}")
            raise

    def validate_api_key(self) -> bool:
        """Validate Google AI API key."""
        return bool(self.api_key)


def create_llm_client(provider: str, api_key: Optional[str] = None) -> BaseLLMClient:
    """Factory function to create an LLM client."""
    providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GoogleAIClient,
    }

    if provider not in providers:
        raise ValueError(
            f"Unsupported provider: {provider}. Choose from: {list(providers.keys())}"
        )

    client_class = providers[provider]
    client = client_class(api_key)

    if not client.validate_api_key():
        raise ValueError(f"Invalid or missing API key for {provider}")

    return client
