# Feature: garak-endpoint-integration, Property 9: Error Response Structure and Safety
"""Property-based tests for error response structure and safety.

Property 9 states:
*For any* error condition that produces an HTTP 4xx or 5xx response, the response body
SHALL conform to the structure {"error": {"type": "<string>", "message": "<string>"}},
and the message field SHALL NOT contain API key values, internal file paths, or Python
stack traces.

**Validates: Requirements 5.1, 5.4, 5.5**
"""

import asyncio
import re
from unittest.mock import MagicMock

import openai
from hypothesis import given, settings
from hypothesis import strategies as st

from api_handlers import error_response, handle_chat_completion, sanitize_message
from api_models import ChatCompletionRequest, ChatMessage


# --- Helpers for detecting sensitive content ---


def contains_file_path(text: str) -> bool:
    """Check if text contains Unix or Windows file paths."""
    # Windows path pattern: C:\something\something
    if re.search(r'[A-Za-z]:\\(?:[^\s\\]+\\)+[^\s\\]*', text):
        return True
    # Unix path pattern: /something/something
    if re.search(r'/(?:[^\s/]+/)+[^\s/]*', text):
        return True
    return False


def contains_api_key(text: str) -> bool:
    """Check if text contains potential API key (20+ alphanumeric chars)."""
    return bool(re.search(r'[A-Za-z0-9_\-]{20,}', text))


def contains_traceback(text: str) -> bool:
    """Check if text contains Python traceback patterns."""
    return bool(re.search(r'File ".*?", line \d+', text))


# --- Strategies ---


unix_path_strategy = st.from_regex(r'/[a-z]+(/[a-z]+){1,4}', fullmatch=True)
windows_path_strategy = st.from_regex(r'[A-Z]:\\[a-z]+(\\[a-z]+){1,3}', fullmatch=True)
api_key_strategy = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    min_size=20,
    max_size=64,
)
traceback_strategy = st.builds(
    lambda fname, lineno: f'File "{fname}", line {lineno}',
    fname=st.from_regex(r'/[a-z]+(/[a-z]+){1,3}\.py', fullmatch=True),
    lineno=st.integers(min_value=1, max_value=999),
)


# --- Property Tests ---


class TestSanitizeMessage:
    """Tests for sanitize_message() function."""

    @settings(max_examples=100, deadline=5000)
    @given(path=unix_path_strategy)
    def test_sanitize_removes_unix_paths(self, path: str):
        """Sanitized output should not contain Unix file paths.

        **Validates: Requirements 5.1, 5.4, 5.5**
        """
        message = f"Error occurred at {path} during processing"
        result = sanitize_message(message)
        assert not contains_file_path(result), (
            f"Sanitized message still contains Unix path: {result}"
        )

    @settings(max_examples=100, deadline=5000)
    @given(path=windows_path_strategy)
    def test_sanitize_removes_windows_paths(self, path: str):
        """Sanitized output should not contain Windows file paths.

        **Validates: Requirements 5.1, 5.4, 5.5**
        """
        message = f"Error occurred at {path} during processing"
        result = sanitize_message(message)
        assert not contains_file_path(result), (
            f"Sanitized message still contains Windows path: {result}"
        )

    @settings(max_examples=100, deadline=5000)
    @given(key=api_key_strategy)
    def test_sanitize_removes_api_keys(self, key: str):
        """Sanitized output should not contain API key-like strings.

        **Validates: Requirements 5.1, 5.4, 5.5**
        """
        message = f"Authentication failed with key {key}"
        result = sanitize_message(message)
        assert not contains_api_key(result), (
            f"Sanitized message still contains API key-like string: {result}"
        )

    @settings(max_examples=100, deadline=5000)
    @given(traceback=traceback_strategy)
    def test_sanitize_removes_tracebacks(self, traceback: str):
        """Sanitized output should not contain Python traceback patterns.

        **Validates: Requirements 5.1, 5.4, 5.5**
        """
        message = f"Exception raised: {traceback}, in module"
        result = sanitize_message(message)
        assert not contains_traceback(result), (
            f"Sanitized message still contains traceback: {result}"
        )


class TestErrorResponse:
    """Tests for error_response() helper function."""

    @settings(max_examples=100, deadline=5000)
    @given(
        status_code=st.sampled_from([400, 401, 403, 404, 422, 500, 502, 503]),
        error_type=st.text(min_size=1, max_size=50, alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_")),
        message=st.text(min_size=1, max_size=200),
    )
    def test_error_response_structure(self, status_code: int, error_type: str, message: str):
        """error_response() should always return correct structure.

        **Validates: Requirements 5.1, 5.4, 5.5**
        """
        result = error_response(status_code, error_type, message)

        # Should be a tuple of (int, dict)
        assert isinstance(result, tuple), "error_response must return a tuple"
        assert len(result) == 2, "error_response tuple must have exactly 2 elements"

        code, body = result
        assert code == status_code, "Status code must match input"
        assert isinstance(body, dict), "Body must be a dict"
        assert "error" in body, "Body must contain 'error' key"
        assert isinstance(body["error"], dict), "'error' value must be a dict"
        assert "type" in body["error"], "Error must contain 'type'"
        assert "message" in body["error"], "Error must contain 'message'"
        assert isinstance(body["error"]["type"], str), "'type' must be a string"
        assert isinstance(body["error"]["message"], str), "'message' must be a string"
        assert body["error"]["type"] == error_type
        assert body["error"]["message"] == message


class TestHandleChatCompletionErrors:
    """Tests for handle_chat_completion() error handling."""

    @settings(max_examples=100, deadline=5000)
    @given(
        exception_type=st.sampled_from([
            "auth_error",
            "connection_error",
            "rate_limit_error",
            "generic_exception",
        ]),
        error_msg=st.text(min_size=1, max_size=100),
    )
    def test_error_structure_on_exceptions(self, exception_type: str, error_msg: str):
        """handle_chat_completion() should return proper error structure on exceptions.

        **Validates: Requirements 5.1, 5.4, 5.5**
        """
        # Build a valid request
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            model="test-model",
        )

        # Create mock rails that raises the specified exception
        mock_rails = MagicMock()
        mock_client = MagicMock()

        if exception_type == "auth_error":
            mock_rails.generate.side_effect = openai.AuthenticationError(
                message="auth failed",
                response=MagicMock(status_code=401),
                body=None,
            )
        elif exception_type == "connection_error":
            mock_rails.generate.side_effect = openai.APIConnectionError(
                request=MagicMock()
            )
        elif exception_type == "rate_limit_error":
            mock_rails.generate.side_effect = openai.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body=None,
            )
        else:
            mock_rails.generate.side_effect = Exception(error_msg)

        # Run the async handler
        result = asyncio.run(
            handle_chat_completion(request, "guarded", mock_rails, mock_client)
        )

        # Should return a tuple (status_code, error_dict)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, "Error tuple must have 2 elements"

        status_code, body = result
        assert isinstance(status_code, int), "Status code must be int"
        assert 400 <= status_code <= 599, f"Status code must be 4xx or 5xx, got {status_code}"

        # Verify error body structure
        assert isinstance(body, dict), "Body must be a dict"
        assert "error" in body, "Body must contain 'error' key"
        assert isinstance(body["error"], dict), "'error' must be a dict"
        assert "type" in body["error"], "Error must contain 'type'"
        assert "message" in body["error"], "Error must contain 'message'"
        assert isinstance(body["error"]["type"], str), "'type' must be a string"
        assert isinstance(body["error"]["message"], str), "'message' must be a string"

        # Verify the message doesn't contain sensitive information
        msg = body["error"]["message"]
        assert not contains_file_path(msg), f"Error message contains file path: {msg}"
        assert not contains_api_key(msg), f"Error message contains API key-like string: {msg}"
        assert not contains_traceback(msg), f"Error message contains traceback: {msg}"
