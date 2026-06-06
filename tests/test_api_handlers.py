"""Property-based tests for api_handlers module.

Tests correctness properties of the request handling logic.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from api_handlers import handle_chat_completion
from api_models import ChatCompletionRequest, ChatMessage


# --- Strategies ---

valid_roles = st.sampled_from(["system", "user", "assistant"])

valid_message_strategy = st.builds(
    ChatMessage,
    role=valid_roles,
    content=st.text(min_size=1, max_size=200),
)

valid_request_strategy = st.builds(
    ChatCompletionRequest,
    messages=st.lists(valid_message_strategy, min_size=1, max_size=10),
    model=st.one_of(st.none(), st.just("test-model")),
)


# --- Property 4: Messages Passthrough Integrity ---
# Feature: garak-endpoint-integration, Property 4: Messages Passthrough Integrity


@settings(max_examples=100, deadline=5000)
@given(request=valid_request_strategy)
def test_messages_passthrough_integrity(request: ChatCompletionRequest):
    """
    **Validates: Requirements 2.5**

    For any valid ChatCompletionRequest with a messages array, the processing
    function SHALL receive the exact same messages (same order, roles, and content)
    without modification.
    """
    # Set up mock rails that captures the messages kwarg
    mock_rails = MagicMock()
    mock_rails.generate = MagicMock(return_value={"content": "test"})

    # Mock direct_client (not used in guarded mode, but required param)
    mock_direct_client = MagicMock()

    # Run handle_chat_completion in guarded mode
    response = asyncio.run(
        handle_chat_completion(
            request=request,
            mode="guarded",
            rails=mock_rails,
            direct_client=mock_direct_client,
        )
    )

    # Assert rails.generate was called exactly once
    mock_rails.generate.assert_called_once()

    # Extract the messages kwarg passed to rails.generate
    call_kwargs = mock_rails.generate.call_args
    passed_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")

    # Assert messages match the original request exactly (same order, roles, content)
    assert len(passed_messages) == len(request.messages), (
        f"Expected {len(request.messages)} messages, got {len(passed_messages)}"
    )

    for i, (original, passed) in enumerate(zip(request.messages, passed_messages)):
        assert passed["role"] == original.role, (
            f"Message {i}: expected role '{original.role}', got '{passed['role']}'"
        )
        assert passed["content"] == original.content, (
            f"Message {i}: expected content '{original.content}', got '{passed['content']}'"
        )
