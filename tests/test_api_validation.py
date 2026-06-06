"""Property-based tests for request validation models.

Tests validate correctness properties from the garak-endpoint-integration design document.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from api_models import ChatCompletionRequest, ChatMessage


# Feature: garak-endpoint-integration, Property 8: Unrecognized Fields Ignored
# **Validates: Requirements 4.8**


# Strategy: generate valid roles
valid_roles = st.sampled_from(["system", "user", "assistant"])

# Strategy: generate a valid ChatMessage as a dict
valid_message_dict = st.fixed_dictionaries(
    {"role": valid_roles, "content": st.text(min_size=1, max_size=200)}
)

# Strategy: generate a non-empty list of valid message dicts
valid_messages = st.lists(valid_message_dict, min_size=1, max_size=5)

# Strategy: generate extra fields with keys that don't collide with known fields
known_fields = {
    "messages",
    "model",
    "temperature",
    "max_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
}

# Generate random key-value pairs for unrecognized fields
extra_field_keys = st.text(min_size=1, max_size=50).filter(lambda k: k not in known_fields)
extra_field_values = st.one_of(
    st.text(max_size=100),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
    st.lists(st.integers(min_value=-100, max_value=100), max_size=5),
)
extra_fields_strategy = st.dictionaries(
    keys=extra_field_keys, values=extra_field_values, min_size=1, max_size=10
)


@given(messages=valid_messages, extra_fields=extra_fields_strategy)
@settings(max_examples=100, deadline=5000)
def test_unrecognized_fields_ignored(messages, extra_fields):
    """Property 8: Unrecognized fields in the request payload are silently ignored.

    For any valid ChatCompletionRequest payload that additionally contains
    arbitrary unrecognized key-value pairs, the model SHALL process the request
    identically to the same request without those extra fields.
    """
    # Build payload with valid fields + extra unrecognized fields
    payload = {"messages": messages, **extra_fields}

    # Construct the model — should succeed despite extra fields
    request_model = ChatCompletionRequest(**payload)

    # Verify the model was created successfully
    assert request_model.messages is not None
    assert len(request_model.messages) == len(messages)

    # Verify each message was parsed correctly
    for i, msg in enumerate(messages):
        assert request_model.messages[i].role == msg["role"]
        assert request_model.messages[i].content == msg["content"]

    # Verify extra fields are NOT present in the resulting model
    model_dict = request_model.dict()
    for key in extra_fields:
        assert key not in model_dict, (
            f"Unrecognized field '{key}' should not be present in the model"
        )

    # Also build the same request WITHOUT extra fields and verify identical result
    clean_payload = {"messages": messages}
    clean_model = ChatCompletionRequest(**clean_payload)

    assert clean_model.dict() == request_model.dict(), (
        "Model with extra fields should produce identical result to model without them"
    )


# Feature: garak-endpoint-integration, Property 7: Optional Field Range Validation
# **Validates: Requirements 4.6, 4.7**

import pytest
from pydantic import ValidationError


# --- Strategies for Property 7 ---

valid_message_strategy = st.builds(
    ChatMessage,
    role=valid_roles,
    content=st.text(min_size=1, max_size=200),
)
valid_messages_list = st.lists(valid_message_strategy, min_size=1, max_size=5)

# In-range strategies for each numeric field
in_range_temperature = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
in_range_max_tokens = st.integers(min_value=1, max_value=4096)
in_range_top_p = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
in_range_frequency_penalty = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
in_range_presence_penalty = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)

# Out-of-range strategies for each numeric field
out_of_range_temperature = st.one_of(
    st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=2.01, allow_nan=False, allow_infinity=False),
)
out_of_range_max_tokens = st.one_of(
    st.integers(max_value=0),
    st.integers(min_value=4097),
)
out_of_range_top_p = st.one_of(
    st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.01, allow_nan=False, allow_infinity=False),
)
out_of_range_frequency_penalty = st.one_of(
    st.floats(max_value=-2.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=2.01, allow_nan=False, allow_infinity=False),
)
out_of_range_presence_penalty = st.one_of(
    st.floats(max_value=-2.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=2.01, allow_nan=False, allow_infinity=False),
)


# --- Property 7 (positive): In-range values are accepted ---


@given(
    messages=valid_messages_list,
    temperature=st.one_of(st.none(), in_range_temperature),
    max_tokens=st.one_of(st.none(), in_range_max_tokens),
    top_p=st.one_of(st.none(), in_range_top_p),
    frequency_penalty=st.one_of(st.none(), in_range_frequency_penalty),
    presence_penalty=st.one_of(st.none(), in_range_presence_penalty),
)
@settings(max_examples=100, deadline=5000)
def test_property7_in_range_values_accepted(
    messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty
):
    """Property 7 (positive): When all present optional numeric fields are within
    their valid ranges, the model SHALL accept the request without validation error.
    """
    data = {"messages": [{"role": m.role, "content": m.content} for m in messages]}
    if temperature is not None:
        data["temperature"] = temperature
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if top_p is not None:
        data["top_p"] = top_p
    if frequency_penalty is not None:
        data["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        data["presence_penalty"] = presence_penalty

    # Should not raise ValidationError
    request = ChatCompletionRequest(**data)
    assert request.messages == messages
    if temperature is not None:
        assert request.temperature == temperature
    if max_tokens is not None:
        assert request.max_tokens == max_tokens
    if top_p is not None:
        assert request.top_p == top_p
    if frequency_penalty is not None:
        assert request.frequency_penalty == frequency_penalty
    if presence_penalty is not None:
        assert request.presence_penalty == presence_penalty


# --- Property 7 (negative): Out-of-range values are rejected ---


@given(
    messages=valid_messages_list,
    temperature=out_of_range_temperature,
)
@settings(max_examples=100, deadline=5000)
def test_property7_out_of_range_temperature_rejected(messages, temperature):
    """Property 7 (negative): temperature outside [0.0, 2.0] SHALL raise ValidationError."""
    data = {
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
    }
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(**data)
    error_fields = [e["loc"][-1] for e in exc_info.value.errors()]
    assert "temperature" in error_fields


@given(
    messages=valid_messages_list,
    max_tokens=out_of_range_max_tokens,
)
@settings(max_examples=100, deadline=5000)
def test_property7_out_of_range_max_tokens_rejected(messages, max_tokens):
    """Property 7 (negative): max_tokens outside [1, 4096] SHALL raise ValidationError."""
    data = {
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "max_tokens": max_tokens,
    }
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(**data)
    error_fields = [e["loc"][-1] for e in exc_info.value.errors()]
    assert "max_tokens" in error_fields


@given(
    messages=valid_messages_list,
    top_p=out_of_range_top_p,
)
@settings(max_examples=100, deadline=5000)
def test_property7_out_of_range_top_p_rejected(messages, top_p):
    """Property 7 (negative): top_p outside [0.0, 1.0] SHALL raise ValidationError."""
    data = {
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "top_p": top_p,
    }
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(**data)
    error_fields = [e["loc"][-1] for e in exc_info.value.errors()]
    assert "top_p" in error_fields


@given(
    messages=valid_messages_list,
    frequency_penalty=out_of_range_frequency_penalty,
)
@settings(max_examples=100, deadline=5000)
def test_property7_out_of_range_frequency_penalty_rejected(messages, frequency_penalty):
    """Property 7 (negative): frequency_penalty outside [-2.0, 2.0] SHALL raise ValidationError."""
    data = {
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "frequency_penalty": frequency_penalty,
    }
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(**data)
    error_fields = [e["loc"][-1] for e in exc_info.value.errors()]
    assert "frequency_penalty" in error_fields


@given(
    messages=valid_messages_list,
    presence_penalty=out_of_range_presence_penalty,
)
@settings(max_examples=100, deadline=5000)
def test_property7_out_of_range_presence_penalty_rejected(messages, presence_penalty):
    """Property 7 (negative): presence_penalty outside [-2.0, 2.0] SHALL raise ValidationError."""
    data = {
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "presence_penalty": presence_penalty,
    }
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(**data)
    error_fields = [e["loc"][-1] for e in exc_info.value.errors()]
    assert "presence_penalty" in error_fields


# Feature: garak-endpoint-integration, Property 6: Invalid Message Detection
# **Validates: Requirements 4.4, 4.5**

import json

from api_server import _validate_messages


# --- Strategies for Property 6 ---

# Strategy to generate invalid roles (not in system/user/assistant)
invalid_role_strategy = st.text(min_size=1, max_size=50).filter(
    lambda r: r not in ("system", "user", "assistant")
)


@given(
    valid_messages=valid_messages,
    insert_index=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=100, deadline=5000)
def test_property6_missing_role_field_detected(valid_messages, insert_index):
    """Property 6a: A message missing the 'role' field at index i SHALL return
    HTTP 422 with an error message identifying index i and the missing role field.
    """
    # Clamp insert index to valid range
    i = insert_index % (len(valid_messages) + 1)

    # Create invalid message: missing 'role'
    invalid_msg = {"content": "some content"}

    # Insert at position i
    messages = list(valid_messages)
    messages.insert(i, invalid_msg)

    # Call _validate_messages
    result = _validate_messages({"messages": messages})

    # Must return a JSONResponse (not None)
    assert result is not None, f"Expected validation error for missing role at index {i}"

    # Must be HTTP 422
    assert result.status_code == 422, f"Expected 422, got {result.status_code}"

    # Parse response body
    body = json.loads(result.body.decode())
    error_message = body["error"]["message"]

    # Must mention the index
    assert str(i) in error_message, (
        f"Error message should contain index {i}, got: {error_message}"
    )

    # Must describe the nature of the failure (missing role)
    assert "role" in error_message.lower(), (
        f"Error message should mention 'role', got: {error_message}"
    )


@given(
    valid_messages=valid_messages,
    insert_index=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=100, deadline=5000)
def test_property6_missing_content_field_detected(valid_messages, insert_index):
    """Property 6b: A message missing the 'content' field at index i SHALL return
    HTTP 422 with an error message identifying index i and the missing content field.
    """
    # Clamp insert index to valid range
    i = insert_index % (len(valid_messages) + 1)

    # Create invalid message: missing 'content'
    invalid_msg = {"role": "user"}

    # Insert at position i
    messages = list(valid_messages)
    messages.insert(i, invalid_msg)

    # Call _validate_messages
    result = _validate_messages({"messages": messages})

    # Must return a JSONResponse (not None)
    assert result is not None, f"Expected validation error for missing content at index {i}"

    # Must be HTTP 422
    assert result.status_code == 422, f"Expected 422, got {result.status_code}"

    # Parse response body
    body = json.loads(result.body.decode())
    error_message = body["error"]["message"]

    # Must mention the index
    assert str(i) in error_message, (
        f"Error message should contain index {i}, got: {error_message}"
    )

    # Must describe the nature of the failure (missing content)
    assert "content" in error_message.lower(), (
        f"Error message should mention 'content', got: {error_message}"
    )


@given(
    valid_messages=valid_messages,
    insert_index=st.integers(min_value=0, max_value=100),
    invalid_role=invalid_role_strategy,
)
@settings(max_examples=100, deadline=5000)
def test_property6_invalid_role_value_detected(valid_messages, insert_index, invalid_role):
    """Property 6c: A message with an invalid role value at index i SHALL return
    HTTP 422 with an error message identifying index i and the invalid role.
    """
    # Clamp insert index to valid range
    i = insert_index % (len(valid_messages) + 1)

    # Create invalid message: invalid role value
    invalid_msg = {"role": invalid_role, "content": "some content"}

    # Insert at position i
    messages = list(valid_messages)
    messages.insert(i, invalid_msg)

    # Call _validate_messages
    result = _validate_messages({"messages": messages})

    # Must return a JSONResponse (not None)
    assert result is not None, f"Expected validation error for invalid role at index {i}"

    # Must be HTTP 422
    assert result.status_code == 422, f"Expected 422, got {result.status_code}"

    # Parse response body
    body = json.loads(result.body.decode())
    error_message = body["error"]["message"]

    # Must mention the index
    assert str(i) in error_message, (
        f"Error message should contain index {i}, got: {error_message}"
    )

    # Must describe the nature of the failure (invalid role)
    assert "role" in error_message.lower(), (
        f"Error message should mention 'role', got: {error_message}"
    )
