"""
api_client.py
-------------
Thin wrapper around the OpenAI-compatible vLLM endpoint.
All network/model configuration lives here.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import os

from openai import OpenAI

# ---------------------------------------------------------------------------
# Server / model configuration
# (Override via env vars VLLM_BASE_URL and MODEL_NAME)
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://172.22.225.5:8013/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-27B")
API_KEY       = "EMPTY"       # vLLM accepts any non-empty string

# Generation defaults (override via call_model kwargs)
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS  = 512

# Retry settings
MAX_RETRIES   = 3
RETRY_DELAY_S = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def img_to_b64(path: Path | str) -> str:
    """Read an image file and return its base64-encoded content."""
    return base64.b64encode(Path(path).read_bytes()).decode()


def _build_client() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)


# ---------------------------------------------------------------------------
# Main call
# ---------------------------------------------------------------------------

def call_model(
    system_prompt: str,
    user_content: list[dict],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_thinking: bool = False,
) -> str:
    """
    Send a chat-completion request to the vLLM server and return the
    assistant's response text.

    Args:
        system_prompt   : System-turn text.
        user_content    : List of content blocks (text / image_url dicts).
        temperature     : Sampling temperature (0.0 = greedy).
        max_tokens      : Maximum tokens to generate.
        enable_thinking : Set True to activate Qwen3's <think> chain-of-thought
                          mode (passes extra_body flag).  The raw <think>…</think>
                          block is stripped from the returned string so callers
                          always receive the final answer only.

    Returns:
        The model's response, stripped of any <think> block.
    """
    client = _build_client()

    # Always pass enable_thinking explicitly.  Qwen3 defaults to thinking mode
    # when this flag is absent, which consumes the entire token budget inside
    # <think>…</think> and leaves an empty final answer.
    extra_body: dict = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_content},
    ]

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            text = response.choices[0].message.content or ""
            return _strip_thinking(text).strip()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

    raise RuntimeError(
        f"Model call failed after {MAX_RETRIES} attempts: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks produced in Qwen3 thinking mode."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
