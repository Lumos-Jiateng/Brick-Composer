"""
api_client.py
-------------
Thin wrapper around the OpenAI-compatible vLLM endpoint.

Module-level defaults can be overridden per-call via base_url / model_name
keyword arguments to call_model(), which is used by evaluate_assembly.py to
support multiple model endpoints at runtime.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Server / model configuration (defaults — overridable per call)
# ---------------------------------------------------------------------------

VLLM_BASE_URL = "http://172.22.225.5:8016/v1"
MODEL_NAME    = "qwen-3-vl"   # name as registered by the vLLM server
API_KEY       = "EMPTY"       # vLLM accepts any non-empty string

# Generation defaults (override via call_model kwargs)
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS  = 1024   # larger budget for multi-brick pose output

# Retry settings
MAX_RETRIES   = 3
RETRY_DELAY_S = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def img_to_b64(path: Path | str) -> str:
    """Read an image file and return its base64-encoded content."""
    return base64.b64encode(Path(path).read_bytes()).decode()


def img_bytes_to_b64(data: bytes) -> str:
    """Encode raw bytes as base64."""
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# Main call
# ---------------------------------------------------------------------------

def call_model(
    system_prompt: str,
    user_content: list[dict],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_thinking: bool = False,
    base_url: str | None = None,
    model_name: str | None = None,
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
        base_url        : Override module-level VLLM_BASE_URL for this call.
        model_name      : Override module-level MODEL_NAME for this call.

    Returns:
        The model's response, stripped of any <think> block.
    """
    effective_url   = base_url   or VLLM_BASE_URL
    effective_model = model_name or MODEL_NAME
    client = OpenAI(base_url=effective_url, api_key=API_KEY)

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
                model=effective_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            text = response.choices[0].message.content or ""
            return _strip_thinking(text).strip()
        except Exception as exc:
            last_exc = exc
            # 400 errors (e.g. context-length exceeded) are not transient —
            # retrying wastes time and will always fail, so bail out immediately.
            exc_str = str(exc)
            if "400" in exc_str or "Bad Request" in exc_str.lower():
                break
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
