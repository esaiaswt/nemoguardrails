"""
Property-based test for message formatting in guarded response.

Feature: annoy-fastembed-rag, Property: Message formatting for Colang 2.0

For any message processed through the guardrails path, the rails.generate() call SHALL
pass only the latest user message (Colang 2.0 requirement) as the messages parameter.

Validates: Requirements 7.2
"""

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st_hyp


@given(message=st_hyp.text(min_size=1))
@settings(max_examples=100)
def test_log_options_inclusion(message: str):
    """
    Property: Message formatting for Colang 2.0

    For any random message string, when processed through the guarded path,
    the rails.generate() call passes the latest user message as the messages parameter.
    Colang 2.0 only supports user messages, so full history is not passed.

    **Validates: Requirements 7.2**
    """
    from app import generate_guarded_response

    # Create a mock LLMRails instance
    mock_rails = MagicMock()
    mock_rails.generate.return_value = {
        "content": "mocked response",
    }

    # Build messages list as the app would
    messages = [{"role": "user", "content": message}]

    # Call the function under test
    generate_guarded_response(mock_rails, messages)

    # Assert that rails.generate was called
    mock_rails.generate.assert_called_once()
    call_kwargs = mock_rails.generate.call_args

    # Verify messages parameter is present and correct
    assert "messages" in call_kwargs.kwargs, (
        f"messages parameter must be passed to rails.generate(). "
        f"Called with args={call_kwargs.args}, kwargs={call_kwargs.kwargs}"
    )

    actual_messages = call_kwargs.kwargs["messages"]

    # Verify the messages contain the user message
    assert len(actual_messages) == 1
    assert actual_messages[0]["role"] == "user"
    assert actual_messages[0]["content"] == message
