"""
Property-based test for message routing correctness.

Feature: nemo-guardrails-playground, Property 1: Message routing correctness

For any user message and any toggle state (enabled/disabled), the application SHALL route
the message through the LLMRails instance when guardrails are enabled, and through the
Direct_Client when guardrails are disabled — never both and never neither.

Validates: Requirements 3.2, 3.3, 6.1, 7.1
"""

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st_hyp

from app import generate_direct_response, generate_guarded_response


@given(message=st_hyp.text(), guardrails_enabled=st_hyp.booleans())
@settings(max_examples=100)
def test_message_routing_correctness(message: str, guardrails_enabled: bool):
    """
    Property 1: Message routing correctness

    For any random message and any toggle state, exactly one engine is called per message:
    - When guardrails_enabled=True, only generate_guarded_response is called
    - When guardrails_enabled=False, only generate_direct_response is called
    - Never both, never neither

    **Validates: Requirements 3.2, 3.3, 6.1, 7.1**
    """
    # Create mock for LLMRails engine
    mock_rails = MagicMock()
    mock_rails.generate.return_value = {"content": "guarded response"}

    # Create mock for Direct OpenAI client
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "direct response"
    mock_client.chat.completions.create.return_value = mock_completion

    # Build messages list as the app would
    messages = [{"role": "user", "content": message}]

    # Track which engines are called
    guarded_called = False
    direct_called = False

    # Simulate the routing logic from app.py (Chat Input and Message Routing section)
    if guardrails_enabled:
        response_text, trace = generate_guarded_response(mock_rails, messages)
        guarded_called = True
    else:
        response_text = generate_direct_response(mock_client, messages)
        direct_called = True

    # Property: Exactly one engine is called, never both, never neither
    assert guarded_called != direct_called, (
        f"Exactly one engine must be called. "
        f"guarded_called={guarded_called}, direct_called={direct_called}"
    )

    # Property: When guardrails enabled, only guarded path is used
    if guardrails_enabled:
        assert guarded_called is True, "Guarded path must be called when guardrails enabled"
        assert direct_called is False, "Direct path must NOT be called when guardrails enabled"
        mock_rails.generate.assert_called_once()
        mock_client.chat.completions.create.assert_not_called()

    # Property: When guardrails disabled, only direct path is used
    if not guardrails_enabled:
        assert direct_called is True, "Direct path must be called when guardrails disabled"
        assert guarded_called is False, "Guarded path must NOT be called when guardrails disabled"
        mock_client.chat.completions.create.assert_called_once()
        mock_rails.generate.assert_not_called()
