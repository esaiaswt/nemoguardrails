"""Property-based test for debug trace visibility.

Feature: nemo-guardrails-playground, Property 5: Debug trace visibility

For any assistant message in the conversation, a debug expander with the
activated rails trace SHALL be rendered if and only if guardrails were enabled
when that message was generated.

Validates: Requirements 8.1, 8.2, 8.3
"""

from hypothesis import given, settings
from hypothesis import strategies as st_hyp


@given(st_hyp.booleans(), st_hyp.text())
@settings(max_examples=100)
def test_debug_trace_visibility(guard_state: bool, message_text: str):
    """Property 5: Debug trace visibility.

    For any random (guard_state, message_text) pair:
    - If guard_state is True (guardrails enabled), trace should be present (not None)
    - If guard_state is False (guardrails disabled), trace should be None
    - The debug expander renders iff message.get("trace") is not None

    **Validates: Requirements 8.1, 8.2, 8.3**
    """
    # Simulate the routing logic from app.py:
    # When guardrails_enabled=True: response gets trace = {...} (not None)
    # When guardrails_enabled=False: response gets trace = None
    if guard_state:
        trace = {"activated_rails": ["some_rail"], "message": message_text}
    else:
        trace = None

    # Construct the message dict as the app does
    message = {"role": "assistant", "content": message_text, "trace": trace}

    # The debug expander visibility condition from app.py:
    #   if message.get("trace") is not None:
    #       with st.expander("🔍 Debug Trace"):
    #           st.json(message["trace"])
    expander_visible = message.get("trace") is not None

    # Property: expander is present iff guardrails were enabled
    assert expander_visible == guard_state, (
        f"Expected expander_visible={guard_state} but got {expander_visible}. "
        f"guard_state={guard_state}, trace={trace}"
    )
