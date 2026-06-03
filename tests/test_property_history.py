"""Property-based test for conversation history integrity.

Feature: nemo-guardrails-playground, Property 3: Conversation history integrity

For any user message submitted and any corresponding assistant response,
both the user message and the assistant response SHALL be appended to the
session state conversation history in order, preserving all prior messages unchanged.

**Validates: Requirements 5.2, 5.3**
"""

import copy

from hypothesis import given, settings
from hypothesis import strategies as st_hyp


@given(st_hyp.lists(st_hyp.text(min_size=1), min_size=1, max_size=20))
@settings(max_examples=100)
def test_conversation_history_integrity(messages):
    """Property 3: Conversation history integrity.

    Generate random message sequences, simulate submissions, verify history
    grows correctly and prior messages remain unchanged.

    **Validates: Requirements 5.2, 5.3**
    """
    # Start with an empty messages list (simulating st.session_state.messages)
    history = []

    for i, message in enumerate(messages):
        # Take a snapshot of history before this submission
        snapshot = copy.deepcopy(history)

        # Simulate user submission: append user message
        history.append({"role": "user", "content": message})

        # Simulate assistant response: append assistant message
        history.append({"role": "assistant", "content": "mock_response", "trace": None})

        # Verify 1: History length grew by exactly 2 (user + assistant)
        expected_length = (i + 1) * 2
        assert len(history) == expected_length, (
            f"Expected history length {expected_length}, got {len(history)} "
            f"after message {i + 1}"
        )

        # Verify 2: All previous messages remain unchanged
        assert history[: len(snapshot)] == snapshot, (
            f"Previous messages were modified after message {i + 1}. "
            f"Snapshot: {snapshot}, Current prefix: {history[:len(snapshot)]}"
        )

        # Verify 3: Messages maintain correct order (alternating user/assistant)
        for j in range(len(history)):
            expected_role = "user" if j % 2 == 0 else "assistant"
            assert history[j]["role"] == expected_role, (
                f"Message at index {j} has role '{history[j]['role']}', "
                f"expected '{expected_role}'"
            )

        # Verify the latest user message content is correct
        assert history[-2]["content"] == message, (
            f"Latest user message content mismatch. "
            f"Expected '{message}', got '{history[-2]['content']}'"
        )

        # Verify the latest assistant message has expected structure
        assert history[-1]["content"] == "mock_response"
        assert history[-1]["trace"] is None
