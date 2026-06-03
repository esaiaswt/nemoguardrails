"""Unit tests for debug trace format and error handling.

Tests that the trace includes kb_results when RAG retrieval occurs,
and that the system degrades gracefully when embedding or search fails.

Validates: Requirements 7.3, 8.3, 8.4, 8.5, 8.6
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from app import generate_guarded_response, _extract_kb_results


def _get_mock_event_loop():
    """Get the mocked get_or_create_event_loop from sys.modules."""
    return sys.modules["nemoguardrails.utils"].get_or_create_event_loop


class TestTraceIncludesKbResults:
    """Tests for kb_results presence in the debug trace."""

    def test_trace_includes_kb_results(self):
        """When RAG retrieval returns results, trace dict contains kb_results list
        with title/score/text entries.

        Validates: Requirement 7.3
        """
        mock_rails = MagicMock()
        mock_rails.generate.return_value = {
            "content": "Here is your answer based on knowledge base.",
        }

        # Set up KB with index that has results
        mock_index = MagicMock()
        mock_index._index = MagicMock()
        mock_index._index.get_nns_by_vector.return_value = (
            [0, 1],  # indices
            [0.3, 0.6],  # distances (angular)
        )

        # Create mock items matching the indices
        mock_item_0 = MagicMock()
        mock_item_0.meta = {"title": "Order Tracking", "body": "Track your order here."}
        mock_item_0.text = "# Order Tracking\n\nTrack your order here."

        mock_item_1 = MagicMock()
        mock_item_1.meta = {"title": "Return Policy", "body": "Items may be returned within 30 days."}
        mock_item_1.text = "# Return Policy\n\nItems may be returned within 30 days."

        mock_index._items = [mock_item_0, mock_item_1]

        mock_kb = MagicMock()
        mock_kb.index = mock_index
        mock_rails.kb = mock_kb

        # Configure the mocked event loop
        mock_loop_fn = _get_mock_event_loop()
        mock_loop = MagicMock()
        mock_loop_fn.return_value = mock_loop
        # Embedding call succeeds
        mock_loop.run_until_complete.return_value = [[0.1] * 384]

        response_text, trace = generate_guarded_response(
            mock_rails, [{"role": "user", "content": "How do I track my order?"}]
        )

        assert "kb_results" in trace
        assert isinstance(trace["kb_results"], list)
        assert len(trace["kb_results"]) == 2

        # Verify each entry has expected keys
        for entry in trace["kb_results"]:
            assert "title" in entry
            assert "score" in entry
            assert "text" in entry

        # Verify specific values
        assert trace["kb_results"][0]["title"] == "Order Tracking"
        assert trace["kb_results"][0]["text"] == "Track your order here."
        # Score = 1 - distance/2 = 1 - 0.3/2 = 0.85
        assert trace["kb_results"][0]["score"] == 0.85

    def test_trace_kb_results_empty_when_no_kb(self):
        """When rails.kb is None, trace has kb_results as empty list.

        Validates: Requirement 7.3
        """
        mock_rails = MagicMock()
        mock_rails.generate.return_value = {
            "content": "I can help with general questions.",
        }
        mock_rails.kb = None

        response_text, trace = generate_guarded_response(
            mock_rails, [{"role": "user", "content": "hello"}]
        )

        assert "kb_results" in trace
        assert trace["kb_results"] == []


class TestGracefulDegradationEmbedding:
    """Tests for graceful degradation when embedding generation fails."""

    def test_embedding_failure_returns_error_to_user(self):
        """When embedding generation raises an exception during trace extraction,
        rails.generate() still runs normally. The trace includes rag_fallback indicator.

        Validates: Requirements 8.3, 8.6
        """
        mock_rails = MagicMock()
        mock_rails.generate.return_value = {
            "content": "I can help you with that.",
        }

        # Set up KB with an index that will fail during embedding
        mock_index = MagicMock()
        mock_index._index = MagicMock()

        mock_kb = MagicMock()
        mock_kb.index = mock_index
        mock_rails.kb = mock_kb

        # Configure the mocked event loop to simulate embedding failure
        mock_loop_fn = _get_mock_event_loop()
        mock_loop = MagicMock()
        mock_loop_fn.return_value = mock_loop
        mock_loop.run_until_complete.side_effect = RuntimeError(
            "FastEmbed model failed to encode"
        )

        response_text, trace = generate_guarded_response(
            mock_rails, [{"role": "user", "content": "What is your return policy?"}]
        )

        # rails.generate() IS called (KB extraction happens after for trace only)
        mock_rails.generate.assert_called_once()

        # Response comes from rails.generate() normally
        assert response_text == "I can help you with that."

        # Trace should indicate rag_fallback and error type (Requirement 8.6)
        assert trace["rag_fallback"] is True
        assert trace["rag_error"] == "embedding_failure"
        assert trace["kb_results"] == []

    def test_extract_kb_results_embedding_failure_returns_error_dict(self):
        """_extract_kb_results returns error info when embedding fails.

        Validates: Requirement 8.3
        """
        mock_rails = MagicMock()
        mock_index = MagicMock()
        mock_index._index = MagicMock()

        mock_kb = MagicMock()
        mock_kb.index = mock_index
        mock_rails.kb = mock_kb

        # Configure the mocked event loop
        mock_loop_fn = _get_mock_event_loop()
        mock_loop = MagicMock()
        mock_loop_fn.return_value = mock_loop
        mock_loop.run_until_complete.side_effect = RuntimeError("model error")

        result = _extract_kb_results(mock_rails, "test query")

        assert result["error"] == "embedding_failure"
        assert result["error_message"] == "model error"
        assert result["results"] == []


class TestGracefulDegradationSearch:
    """Tests for graceful degradation when Annoy index search fails."""

    def test_search_failure_continues_with_fallback(self):
        """When Annoy index search raises an exception, the function still
        returns a normal LLM response but trace includes rag_fallback indicator.

        Validates: Requirements 8.4, 8.6
        """
        mock_rails = MagicMock()
        mock_rails.generate.return_value = {
            "content": "Let me help you with that.",
        }

        # Set up KB with an index where search fails
        mock_index = MagicMock()
        mock_index._index = MagicMock()
        # Embedding succeeds but search fails
        mock_index._index.get_nns_by_vector.side_effect = RuntimeError(
            "Annoy index corrupted"
        )

        mock_kb = MagicMock()
        mock_kb.index = mock_index
        mock_rails.kb = mock_kb

        # Configure the mocked event loop - embedding succeeds
        mock_loop_fn = _get_mock_event_loop()
        mock_loop = MagicMock()
        mock_loop_fn.return_value = mock_loop
        mock_loop.run_until_complete.return_value = [[0.1] * 384]

        response_text, trace = generate_guarded_response(
            mock_rails, [{"role": "user", "content": "Tell me about shipping"}]
        )

        # Should still return a valid LLM response (Requirement 8.4)
        assert response_text == "Let me help you with that."

        # Trace should include fallback indicators (Requirement 8.6)
        assert trace["rag_fallback"] is True
        assert trace["rag_error"] == "search_failure"
        assert trace["kb_results"] == []

        # rails.generate was still called (graceful degradation)
        mock_rails.generate.assert_called_once()

    def test_extract_kb_results_search_failure_returns_error_dict(self):
        """_extract_kb_results returns error info when search fails.

        Validates: Requirement 8.4
        """
        mock_rails = MagicMock()
        mock_index = MagicMock()
        mock_index._index = MagicMock()
        mock_index._index.get_nns_by_vector.side_effect = RuntimeError("corrupted")

        mock_kb = MagicMock()
        mock_kb.index = mock_index
        mock_rails.kb = mock_kb

        # Configure the mocked event loop - embedding succeeds
        mock_loop_fn = _get_mock_event_loop()
        mock_loop = MagicMock()
        mock_loop_fn.return_value = mock_loop
        mock_loop.run_until_complete.return_value = [[0.1] * 384]

        result = _extract_kb_results(mock_rails, "test query")

        assert result["error"] == "search_failure"
        assert result["error_message"] == "corrupted"
        assert result["results"] == []


class TestCascadingFailure:
    """Tests for cascading failure (both RAG and LLM fail)."""

    def test_cascading_failure_rag_and_llm(self):
        """When rails.generate() fails, returns explicit error message to user.

        Validates: Requirement 8.5
        """
        mock_rails = MagicMock()
        # LLM generation fails
        mock_rails.generate.side_effect = RuntimeError("LLM API unreachable")

        mock_rails.kb = None

        response_text, trace = generate_guarded_response(
            mock_rails, [{"role": "user", "content": "What is your policy?"}]
        )

        # Should return explicit error message (Requirement 8.5)
        assert "unable to generate a response" in response_text
        assert "system error" in response_text

        # Trace should indicate error
        assert trace["status"] == "error"
        assert trace["kb_results"] == []

    def test_llm_failure_without_rag_issue(self):
        """When rails.generate() fails but RAG was fine (no KB),
        still returns error message.

        Validates: Requirement 8.5
        """
        mock_rails = MagicMock()
        mock_rails.generate.side_effect = RuntimeError("LLM timeout")
        mock_rails.kb = None

        response_text, trace = generate_guarded_response(
            mock_rails, [{"role": "user", "content": "hello"}]
        )

        # Should return explicit error message
        assert "unable to generate a response" in response_text
        assert trace["status"] == "error"
        assert trace["kb_results"] == []
