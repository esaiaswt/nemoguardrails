"""Request handling logic for the OpenAI-compatible Chat Completions API.

Bridges between the FastAPI API layer and the guardrails/direct processing backends,
formatting responses in the OpenAI Chat Completions response schema.
"""

import asyncio
import logging
import os
import re
import time
from collections import deque
from typing import Any, Optional, Union
from uuid import uuid4

import openai

from api_config import resolve_guardrails_mode
from api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ResponseMessage,
    Usage,
)

try:
    from nemoguardrails import LLMRails
except ImportError:
    LLMRails = Any  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = Any  # type: ignore


logger = logging.getLogger("api_server.handlers")

DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"

# Timeout for upstream requests (seconds). Prevents infinite hangs when NIM is degraded.
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT", "60"))

# Rate limiter: NVIDIA NIM allows 40 API calls per minute.
# We track call timestamps and block if we'd exceed the limit.
_RATE_LIMIT_MAX_CALLS = int(os.environ.get("NIM_RATE_LIMIT", "40"))
_RATE_LIMIT_WINDOW_SECONDS = 60
_call_timestamps: deque = deque()


def _rate_limit_check() -> Optional[float]:
    """Check if we're within the rate limit. Returns wait time if throttled, None if OK."""
    now = time.time()
    # Evict timestamps older than the window
    while _call_timestamps and _call_timestamps[0] < now - _RATE_LIMIT_WINDOW_SECONDS:
        _call_timestamps.popleft()
    if len(_call_timestamps) >= _RATE_LIMIT_MAX_CALLS:
        # Calculate how long to wait until the oldest call falls out of the window
        wait_time = _call_timestamps[0] + _RATE_LIMIT_WINDOW_SECONDS - now
        return max(wait_time, 0.1)
    return None


def _rate_limit_record():
    """Record a call timestamp for rate limiting."""
    _call_timestamps.append(time.time())


def configure_logging():
    """Configure logging to write to app.log."""
    handler = logging.FileHandler("app.log")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)


def sanitize_message(message: str) -> str:
    """Strip sensitive information from error messages.

    Removes file paths, potential API keys, and Python stack traces
    to prevent leaking internal details in error responses.

    Args:
        message: The raw error message string.

    Returns:
        A sanitized message safe for inclusion in API responses.
    """
    # Strip Python traceback patterns (File "...", line N)
    sanitized = re.sub(r'File ".*?", line \d+', '[traceback redacted]', message)

    # Replace Windows file paths (C:\path\to\file or similar)
    sanitized = re.sub(r'[A-Za-z]:\\(?:[^\s\\]+\\)*[^\s\\]*', '[path redacted]', sanitized)

    # Replace Unix file paths (/path/to/file)
    sanitized = re.sub(r'/(?:[^\s/]+/)+[^\s/]*', '[path redacted]', sanitized)

    # Replace long alphanumeric strings (potential API keys, 20+ chars)
    sanitized = re.sub(r'[A-Za-z0-9_\-]{20,}', '[redacted]', sanitized)

    return sanitized


def error_response(status_code: int, error_type: str, message: str) -> tuple[int, dict]:
    """Build a standardized error response tuple.

    Args:
        status_code: The HTTP status code for the error.
        error_type: The error classification type string.
        message: A human-readable error description.

    Returns:
        A tuple of (status_code, error_dict) matching the ErrorResponse format.
    """
    return (status_code, {
        "error": {
            "type": error_type,
            "message": message,
        }
    })


async def handle_chat_completion(
    request: ChatCompletionRequest,
    mode: str,
    rails,
    direct_client: OpenAI,
) -> Union[ChatCompletionResponse, tuple[int, dict]]:
    """Process a chat completion request through the selected mode.

    Args:
        request: The validated chat completion request.
        mode: The resolved guardrails mode ("guarded" or "unguarded").
        rails: The LLMRails instance for guarded mode processing.
        direct_client: The OpenAI client for unguarded mode (direct passthrough).

    Returns:
        A ChatCompletionResponse with the LLM-generated content, or a tuple
        of (status_code, error_dict) when an error occurs.
    """
    request_id = f"req-{uuid4().hex[:12]}"
    start_time = time.time()

    # INFO log: request metadata (no message content)
    logger.info(
        f"Request {request_id}: mode={mode}, messages={len(request.messages)}"
    )

    # DEBUG log: truncated last user message content (first 100 chars)
    last_user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), None
    )
    if last_user_msg:
        logger.debug(f"Request {request_id}: user_message={last_user_msg[:100]}")
    else:
        logger.debug(f"Request {request_id}: no user message present")

    model_name = request.model or DEFAULT_MODEL

    # Convert messages to list of dicts for processing
    messages_dicts = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    try:
        if mode == "guarded":
            content, prompt_tokens, completion_tokens = await asyncio.wait_for(
                _handle_guarded(messages_dicts, rails),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        else:
            content, prompt_tokens, completion_tokens = await asyncio.wait_for(
                _handle_unguarded(messages_dicts, model_name, request, direct_client),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"Error {request_id}: type=TimeoutError, message=Request timed out after {REQUEST_TIMEOUT_SECONDS}s, duration_ms={duration_ms}"
        )
        return error_response(504, "upstream_timeout", f"Request timed out after {REQUEST_TIMEOUT_SECONDS} seconds. The upstream service may be degraded.")
    except openai.AuthenticationError as exc:
        logger.error(
            f"Error {request_id}: type=AuthenticationError, message={str(exc)}"
        )
        return error_response(502, "upstream_auth_error", "Authentication failed with upstream LLM service.")
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        logger.error(
            f"Error {request_id}: type={type(exc).__name__}, message={str(exc)}"
        )
        return error_response(502, "upstream_connection_error", "Upstream service is unreachable.")
    except openai.RateLimitError as exc:
        logger.error(
            f"Error {request_id}: type=RateLimitError, message={str(exc)}"
        )
        return error_response(502, "upstream_rate_limit", "Upstream service throttled the request.")
    except Exception as exc:
        logger.error(
            f"Error {request_id}: type={type(exc).__name__}, message={str(exc)}"
        )
        return error_response(500, "internal_error", sanitize_message(str(exc)))

    # INFO log: response metadata (no message content)
    duration_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"Response {request_id}: content_length={len(content)}, duration_ms={duration_ms}"
    )

    return _build_response(content, model_name, prompt_tokens, completion_tokens)


async def _handle_guarded(
    messages_dicts: list,
    rails,
) -> tuple[str, int, int]:
    """Process messages through LLMRails (guarded mode).

    Wraps the synchronous rails.generate() call in asyncio.to_thread()
    to avoid blocking the FastAPI event loop. Includes rate limiting
    to stay within NVIDIA NIM's 40 calls/minute cap.

    Returns:
        Tuple of (content, prompt_tokens, completion_tokens).
    """
    # Rate limit check — wait if we'd exceed NIM's limit
    wait_time = _rate_limit_check()
    if wait_time:
        logger.info(f"Rate limit: waiting {wait_time:.1f}s before calling NIM")
        await asyncio.sleep(wait_time)

    _rate_limit_record()
    result = await asyncio.to_thread(rails.generate, messages=messages_dicts)
    content = result.get("content", "")

    # Approximate token counts since LLMRails doesn't provide them
    prompt_text = " ".join(m["content"] for m in messages_dicts)
    prompt_tokens = len(prompt_text) // 4
    completion_tokens = len(content) // 4

    return content, prompt_tokens, completion_tokens


async def _handle_unguarded(
    messages_dicts: list,
    model_name: str,
    request: ChatCompletionRequest,
    direct_client: OpenAI,
) -> tuple[str, int, int]:
    """Process messages through the direct OpenAI client (unguarded mode).

    Includes rate limiting to stay within NVIDIA NIM's 40 calls/minute cap.

    Returns:
        Tuple of (content, prompt_tokens, completion_tokens).
    """
    # Rate limit check — wait if we'd exceed NIM's limit
    wait_time = _rate_limit_check()
    if wait_time:
        logger.info(f"Rate limit: waiting {wait_time:.1f}s before calling NIM")
        await asyncio.sleep(wait_time)

    _rate_limit_record()

    # Build kwargs for the API call
    kwargs = {
        "model": model_name,
        "messages": messages_dicts,
    }
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p

    response = await asyncio.to_thread(
        direct_client.chat.completions.create, **kwargs
    )

    content = response.choices[0].message.content or ""

    # Use usage from the upstream response if available
    if response.usage:
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
    else:
        # Approximate if not available
        prompt_text = " ".join(m["content"] for m in messages_dicts)
        prompt_tokens = len(prompt_text) // 4
        completion_tokens = len(content) // 4

    return content, prompt_tokens, completion_tokens


def _build_response(
    content: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> ChatCompletionResponse:
    """Build a ChatCompletionResponse from the generated content.

    Args:
        content: The assistant's response text.
        model_name: The model name to include in the response.
        prompt_tokens: Number of prompt tokens (actual or approximated).
        completion_tokens: Number of completion tokens (actual or approximated).

    Returns:
        A fully populated ChatCompletionResponse.
    """
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex[:24]}",
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def get_health_status(
    rails: Optional[LLMRails],
    direct_client: Optional[OpenAI],
    default_mode: str,
    model_name: str,
) -> tuple[int, dict]:
    """Return health status and HTTP status code.

    Returns HTTP 200 with status details when both components are initialized,
    or HTTP 503 with reason when a component is unavailable.
    """
    if rails is not None and direct_client is not None:
        return (200, {
            "status": "healthy",
            "guardrails_mode": default_mode,
            "model": model_name,
        })

    # Determine which component(s) are unavailable
    reasons = []
    if rails is None:
        reasons.append("LLMRails not initialized")
    if direct_client is None:
        reasons.append("Direct client not initialized")

    reason = " and ".join(reasons)

    return (503, {
        "status": "unhealthy",
        "reason": reason,
    })
