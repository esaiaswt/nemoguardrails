"""
Integration tests for end-to-end flows.

Tests full chat flow, toggle routing, config update, and error recovery scenarios.

Validates: Requirements 7.1, 7.4, 7.5, 7.7
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

from app import create_rails_instance, generate_direct_response, generate_guarded_response


class TestFullChatFlowWithGuardrails:
    """Test full chat flow: submit message with guardrails enabled, verify response in history."""

    def test_guarded_response_returns_text_and_trace(self):
        """Submit a message through guardrails, verify response text and trace are returned."""
        mock_rails = MagicMock()
        mock_rails.generate.return_value = {
            "content": "I can help you with your order."
        }

        messages = [{"role": "user", "content": "What is my order status?"}]
        response_text, trace = generate_guarded_response(mock_rails, messages)

        assert response_text == "I can help you with your order."
        assert isinstance(trace, dict)

    def test_history_contains_user_and_assistant_messages(self):
        """After a chat exchange, history has both user and assistant messages."""
        mock_rails = MagicMock()
        mock_rails.generate.return_value = {
            "content": "Hello! How can I help you?"
        }

        history = []
        user_msg = {"role": "user", "content": "Hi there"}
        history.append(user_msg)

        messages_for_api = [{"role": m["role"], "content": m["content"]} for m in history]
        response_text, trace = generate_guarded_response(mock_rails, messages_for_api)

        assistant_msg = {"role": "assistant", "content": response_text, "trace": trace}
        history.append(assistant_msg)

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hi there"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hello! How can I help you?"
        assert "trace" in history[1]

    def test_multiple_turns_preserve_history(self):
        """Multiple chat turns build up history correctly."""
        mock_rails = MagicMock()
        responses = ["First response", "Second response", "Third response"]
        mock_rails.generate.side_effect = [
            {"content": r} for r in responses
        ]

        history = []
        user_messages = ["Hello", "How are you?", "Tell me about returns"]

        for i, user_text in enumerate(user_messages):
            history.append({"role": "user", "content": user_text})
            messages_for_api = [{"role": m["role"], "content": m["content"]} for m in history]
            response_text, trace = generate_guarded_response(mock_rails, messages_for_api)
            history.append({"role": "assistant", "content": response_text, "trace": trace})

        assert len(history) == 6  # 3 user + 3 assistant
        assert history[0]["content"] == "Hello"
        assert history[1]["content"] == "First response"
        assert history[4]["content"] == "Tell me about returns"
        assert history[5]["content"] == "Third response"


class TestToggleRoutingChanges:
    """Test toggle behavior: switch toggle and verify routing changes."""

    def test_guardrails_enabled_routes_through_rails(self):
        """With guardrails enabled, messages route through rails.generate."""
        mock_rails = MagicMock()
        mock_rails.generate.return_value = {"content": "Guarded reply"}

        messages = [{"role": "user", "content": "test message"}]
        response_text, trace = generate_guarded_response(mock_rails, messages)

        # The current implementation calls rails.generate(messages=...) without options
        mock_rails.generate.assert_called_once_with(
            messages=messages,
        )
        assert response_text == "Guarded reply"

    def test_guardrails_disabled_routes_through_direct_client(self):
        """With guardrails disabled, messages route through client.chat.completions.create."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Direct reply"
        mock_client.chat.completions.create.return_value = mock_completion

        messages = [{"role": "user", "content": "test message"}]
        response_text = generate_direct_response(mock_client, messages)

        # Verify the correct model and parameters are used
        mock_client.chat.completions.create.assert_called_once_with(
            model="nvidia/llama-3.1-nemotron-nano-8b-v1",
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
        )
        assert response_text == "Direct reply"

    def test_switching_toggle_changes_routing(self):
        """Switching between enabled/disabled routes to different engines."""
        mock_rails = MagicMock()
        mock_rails.generate.return_value = {"content": "Guarded"}

        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Direct"
        mock_client.chat.completions.create.return_value = mock_completion

        messages = [{"role": "user", "content": "same message"}]

        # Guardrails ON
        response_on, _ = generate_guarded_response(mock_rails, messages)
        assert response_on == "Guarded"
        mock_rails.generate.assert_called_once()
        mock_client.chat.completions.create.assert_not_called()

        # Guardrails OFF
        response_off = generate_direct_response(mock_client, messages)
        assert response_off == "Direct"
        mock_client.chat.completions.create.assert_called_once()


class TestConfigUpdateCreatesNewInstance:
    """Test config update: verify create_rails_instance uses RailsConfig.from_path."""

    def test_create_rails_instance_with_valid_config_dir(self):
        """Calling create_rails_instance uses RailsConfig.from_path with CONFIG_DIR."""
        import app

        # The RailsConfig in app module is already a mock from conftest
        mock_config = MagicMock()
        mock_config.docs = []
        app.RailsConfig.from_path.reset_mock()
        app.RailsConfig.from_path.return_value = mock_config

        mock_instance = MagicMock()
        mock_instance.kb = None
        app.LLMRails.reset_mock()
        app.LLMRails.return_value = mock_instance

        result = create_rails_instance()

        app.RailsConfig.from_path.assert_called_once()
        app.LLMRails.assert_called_once_with(mock_config)
        assert result == mock_instance

    def test_new_instance_replaces_old(self):
        """Simulating the update flow: new instance replaces old in session state."""
        import app

        mock_config = MagicMock()
        mock_config.docs = []
        app.RailsConfig.from_path.reset_mock()
        app.RailsConfig.from_path.return_value = mock_config

        old_instance = MagicMock(name="old_rails")
        new_instance = MagicMock(name="new_rails")
        new_instance.kb = None
        app.LLMRails.reset_mock()
        app.LLMRails.return_value = new_instance

        # Simulate session state update flow
        session_rails = old_instance

        result = create_rails_instance()
        session_rails = result

        assert session_rails == new_instance
        assert session_rails != old_instance


class TestErrorRecoveryRetainsPreviousInstance:
    """Test error recovery: submit invalid config, verify app retains previous working instance."""

    @patch("app.RailsConfig.from_path")
    def test_invalid_config_raises_exception(self, mock_from_path):
        """When RailsConfig.from_path raises, create_rails_instance propagates the error."""
        mock_from_path.side_effect = ValueError("Invalid YAML syntax")

        with patch("app.CONFIG_DIR", os.path.dirname(os.path.abspath(__file__))):
            try:
                create_rails_instance()
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Invalid YAML syntax" in str(e)

    @patch("app.RailsConfig.from_path")
    def test_previous_instance_retained_on_failure(self, mock_from_path):
        """Simulating the update flow: exception prevents replacing old instance."""
        mock_from_path.side_effect = Exception("Parse error in configuration")

        # Simulate session state with existing working instance
        old_instance = MagicMock(name="working_rails")
        session_rails = old_instance

        # Simulate the update button handler from app.py
        try:
            new_rails = create_rails_instance()
            session_rails = new_rails  # This line only runs on success
        except Exception:
            pass  # Error caught, old instance retained

        assert session_rails == old_instance

    @patch("app.RailsConfig.from_path")
    def test_app_still_functional_after_failed_update(self, mock_from_path):
        """After a failed config update, the old instance still processes messages."""
        mock_from_path.side_effect = RuntimeError("Bad config")

        old_instance = MagicMock(name="working_rails")
        old_instance.generate.return_value = {"content": "Still working!"}
        session_rails = old_instance

        # Attempt update (fails)
        try:
            new_rails = create_rails_instance()
            session_rails = new_rails
        except Exception:
            pass

        # Old instance still works
        messages = [{"role": "user", "content": "Are you still there?"}]
        response_text, trace = generate_guarded_response(session_rails, messages)

        assert response_text == "Still working!"
        old_instance.generate.assert_called_once()
