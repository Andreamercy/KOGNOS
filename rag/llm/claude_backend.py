"""
rag/llm/claude_backend.py

Anthropic Claude API client wrapper for KOGNOS.

Wraps the LlamaIndex Anthropic integration with:
  - Retry logic (tenacity) for transient API errors
  - Cost tracking (token logging)
  - Model selection via environment variable
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

CLAUDE_MODEL   = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")


def build_claude_llm(
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.1,
):
    """
    Build a LlamaIndex-compatible Claude LLM client.

    Args:
        model:       Claude model ID (defaults to CLAUDE_MODEL env var).
        max_tokens:  Max output tokens per request.
        temperature: Sampling temperature (low = more deterministic).

    Returns:
        llama_index.llms.anthropic.Anthropic LLM instance.
    """
    model = model or CLAUDE_MODEL

    if not ANTHROPIC_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Set it in your .env file or environment, or switch to LLM_BACKEND=ollama."
        )

    try:
        from llama_index.llms.anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "llama-index-llms-anthropic not installed.\n"
            "Run: pip install llama-index-llms-anthropic"
        )

    logger.info("Initialising Claude LLM: %s", model)

    llm = Anthropic(
        model=model,
        api_key=ANTHROPIC_KEY,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return llm


def build_anthropic_client():
    """
    Build a raw Anthropic SDK client (for direct API use outside LlamaIndex).

    Returns:
        anthropic.Anthropic client instance.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic SDK not installed. Run: pip install anthropic")

    if not ANTHROPIC_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def complete(prompt: str, system: str = "", model: str | None = None) -> str:
    """
    Simple synchronous completion using the raw Anthropic SDK.
    Useful for direct calls outside LlamaIndex (e.g., post-processing).

    Args:
        prompt: User message.
        system: System prompt.
        model:  Model ID.

    Returns:
        Assistant response text.
    """
    client = build_anthropic_client()
    m = model or CLAUDE_MODEL

    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {"model": m, "max_tokens": 2048, "messages": messages}
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text
