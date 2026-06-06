"""Property-based tests for API logging behavior.

Tests correctness properties related to logging content exclusion
and truncation as defined in the design document.
"""

import asyncio
import logging
import re
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from api_handlers import handle_chat_completion
from api_models import ChatCompletionRequest, ChatMessage


# Feature: garak-endpoint-integration, Property 11: DEBUG Log Truncation
# **Validates: Requirements 7.6**
@settings(max_examples=100, deadline=5000)
@given(
    long_content=st.text(min_size=101, max_size=500).filter(lambda s: s.strip())
)
def test_debug_log_truncation(long_content: str):
    """Property 11: For any request where the last user message content exceeds
    100 characters, the DEBUG-level log entry SHALL contain at most 100 characters
    of that message content.
    """
    # Create a ChatCompletionRequest with a user message containing long content
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=long_content)]
    )

    # Set up a logging handler to capture DEBUG-level log records
    handler_logger = logging.getLogger("api_server.handlers")
    original_level = handler_logger.level
    handler_logger.setLevel(logging.DEBUG)

    captured_records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_records.append(record)

    capture_handler = CaptureHandler()
    capture_handler.setLevel(logging.DEBUG)
    handler_logger.addHandler(capture_handler)

    try:
        # Mock LLMRails.generate() to return a simple response
        mock_rails = MagicMock()
        mock_rails.generate.return_value = {"content": "test"}

        mock_direct_client = MagicMock()

        # Run the async handler
        asyncio.run(
            handle_chat_completion(
                request=request,
                mode="guarded",
                rails=mock_rails,
                direct_client=mock_direct_client,
            )
        )

        # Find the DEBUG log record that contains "user_message="
        debug_records = [
            r for r in captured_records
            if r.levelno == logging.DEBUG and "user_message=" in r.getMessage()
        ]

        assert len(debug_records) >= 1, (
            "Expected at least one DEBUG log record with 'user_message='"
        )

        for record in debug_records:
            msg = record.getMessage()
            # Extract the user_message portion after "user_message="
            # Use re.DOTALL since truncated content may contain newlines
            match = re.search(r"user_message=(.*)", msg, re.DOTALL)
            assert match is not None, (
                f"Could not parse user_message from log: {msg!r}"
            )
            logged_content = match.group(1)
            assert len(logged_content) <= 100, (
                f"DEBUG log contains {len(logged_content)} chars of user message "
                f"content, expected at most 100. Content: {logged_content!r}"
            )
    finally:
        # Clean up: remove handler and restore level
        handler_logger.removeHandler(capture_handler)
        handler_logger.setLevel(original_level)
        captured_records.clear()
