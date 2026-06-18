"""
rag/llm/ollama_backend.py

Local Ollama LLM client — runs entirely offline with no API key required.

Supported models (pull first with `ollama pull <model>`):
  - llama3         (recommended, good reasoning)
  - mistral        (fast, lower memory)
  - codellama      (good for kubectl command generation)
  - phi3           (lightweight, runs on CPU)

Usage: set LLM_BACKEND=ollama in .env
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")


def build_ollama_llm(
    model: str | None = None,
    base_url: str | None = None,
    request_timeout: float = 120.0,
    temperature: float = 0.1,
):
    """
    Build a LlamaIndex-compatible Ollama LLM client.

    Args:
        model:           Ollama model name (defaults to OLLAMA_MODEL env var).
        base_url:        Ollama server URL.
        request_timeout: HTTP timeout in seconds.
        temperature:     Sampling temperature.

    Returns:
        llama_index.llms.ollama.Ollama LLM instance.
    """
    model   = model   or OLLAMA_MODEL
    url     = base_url or OLLAMA_BASE_URL

    try:
        from llama_index.llms.ollama import Ollama
    except ImportError:
        raise ImportError(
            "llama-index-llms-ollama not installed.\n"
            "Run: pip install llama-index-llms-ollama"
        )

    logger.info("Initialising Ollama LLM: %s @ %s", model, url)

    llm = Ollama(
        model=model,
        base_url=url,
        request_timeout=request_timeout,
        temperature=temperature,
        context_window=4096,
    )
    return llm


def is_ollama_running(base_url: str | None = None) -> bool:
    """
    Quick health check — returns True if the Ollama server is reachable.

    Args:
        base_url: Ollama server URL.

    Returns:
        True if server responds to /api/tags.
    """
    import urllib.request
    url = (base_url or OLLAMA_BASE_URL).rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_available_models(base_url: str | None = None) -> list[str]:
    """
    List model names available in the local Ollama installation.

    Returns:
        List of model name strings, or empty list if Ollama is unreachable.
    """
    import json
    import urllib.request

    url = (base_url or OLLAMA_BASE_URL).rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.warning("Cannot reach Ollama at %s: %s", url, e)
        return []


def complete(prompt: str, system: str = "", model: str | None = None) -> str:
    """
    Simple synchronous Ollama completion (without LlamaIndex).

    Args:
        prompt: User prompt text.
        system: System prompt.
        model:  Model name.

    Returns:
        Generated response text.
    """
    import json
    import urllib.request

    m   = model or OLLAMA_MODEL
    url = OLLAMA_BASE_URL.rstrip("/") + "/api/generate"

    payload = json.dumps({
        "model":  m,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode()

    try:
        req  = urllib.request.Request(url, data=payload,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("response", "")
    except Exception as e:
        logger.error("Ollama completion failed: %s", e)
        return f"Error: {e}"
