"""Unit tests for message processing functions.

Tests guarded response extraction, direct response extraction,
and error handling when API calls fail.

Validates: Requirements 7.2, 7.3, 8.3, 8.4, 8.5
"""

from unittest.mock import MagicMock, patch

import pytest

from app import generate_guarded_response, generate_direct_response


class TestGuardedResponse:
    """Tests for generate_guarded_response function."""

    def test_guarded_response_extracts_content(self):
        """Test that guarded response correctly extracts content from rails result.

        The current implementation includes engine/status/kb_results in the trace.

        Validates: Requirement 7.2, 7.3
        """
        mock_rails = MagicMock()
        mock_rails.generate.return_value = {
            "content": "test response",
            "log": {"activated_rails": {"rail1": True}},
        }

        response_text, trace = generate_guarded_response(
            mock_rails, [{"role": "user", "content": "hello"}]
        )

        assert response_text == "test response"
        # The trace should include standard fields set by generate_guarded_response
        assert "kb_results" in trace
        assert isinstance(trace["kb_results"], list)

    def test_guarded_response_handles_normal_response(self):
        """Test that guarded response includes engine/status for normal responses.

        Validates: Requirement 7.3
        """
        mock_rails = MagicMock()
        mock_rails.generate.return_value = {"content": "response"}

        response_text, trace = generate_guarded_response(
            mock_rails, [{"role": "user", "content": "hello"}]
        )

        assert response_text == "response"
        # Normal response should have engine and status in trace
        assert trace.get("engine") == "guardrails"
        assert trace.get("status") == "completed"

    def test_guarded_response_error_returns_error_message(self):
        """Test that exceptions from rails.generate() are handled gracefully.

        The current implementation catches LLM generation failures and returns
        an error message instead of propagating the exception.

        Validates: Requirement 8.5
        """
        mock_rails = MagicMock()
        mock_rails.generate.side_effect = RuntimeError("Rails generation failed")

        response_text, trace = generate_guarded_response(
            mock_rails, [{"role": "user", "content": "hello"}]
        )

        # Should return an error message to the user, not raise
        assert "unable to generate" in response_text.lower() or "error" in response_text.lower()
        assert trace.get("status") == "error"


class TestDirectResponse:
    """Tests for generate_direct_response function."""

    def test_direct_response_extracts_content(self):
        """Test that direct response correctly extracts content from OpenAI completion.

        Validates: Requirement 7.1, 7.3
        """
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "direct response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        result = generate_direct_response(
            mock_client, [{"role": "user", "content": "hello"}]
        )

        assert result == "direct response"

    def test_direct_response_error_propagates(self):
        """Test that API errors from OpenAI client propagate to caller.

        Validates: Requirement 7.1
        """
        import sys

        mock_client = MagicMock()
        # Use the mocked openai.APIError from conftest (which is just Exception)
        api_error = sys.modules["openai"].APIError("API connection failed")
        mock_client.chat.completions.create.side_effect = api_error

        with pytest.raises(Exception, match="API connection failed"):
            generate_direct_response(
                mock_client, [{"role": "user", "content": "hello"}]
            )
