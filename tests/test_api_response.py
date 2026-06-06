"""Property-based tests for API response behavior."""

import asyncio
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from api_handlers import handle_chat_completion
from api_models import ChatCompletionRequest, ChatMessage


# --- Strategies ---

# Strategy for generating a valid ChatMessage
valid_roles = st.sampled_from(["system", "user", "assistant"])
valid_content = st.text(min_size=1, max_size=200)

valid_message_strategy = st.builds(
    ChatMessage,
    role=valid_roles,
    content=valid_content,
)

# Strategy for generating a valid ChatCompletionRequest
valid_request_strategy = st.builds(
    ChatCompletionRequest,
    messages=st.lists(valid_message_strategy, min_size=1, max_size=5),
    model=st.one_of(st.none(), st.just("nvidia/llama-3.1-nemotron-nano-8b-v1")),
    temperature=st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0)),
    max_tokens=st.one_of(st.none(), st.integers(min_value=1, max_value=4096)),
)


# Feature: garak-endpoint-integration, Property 3: Response ID Uniqueness
# **Validates: Requirements 2.6**


@settings(max_examples=100, deadline=5000)
@given(
    n=st.integers(min_value=2, max_value=20),
    requests=st.lists(valid_request_strategy, min_size=20, max_size=20),
)
def test_response_id_uniqueness(n: int, requests):
    """Property 3: Response ID Uniqueness.

    For any sequence of N valid requests processed by the endpoint,
    all N response `id` values SHALL be distinct strings.
    """
    # Use only the first n requests from the generated list
    requests_to_use = requests[:n]

    # Mock LLMRails.generate() to return {"content": "response"}
    mock_rails = MagicMock()
    mock_rails.generate.return_value = {"content": "response"}

    # Mock direct client (not used in guarded mode, but required param)
    mock_direct_client = MagicMock()

    async def run_requests():
        responses = []
        for req in requests_to_use:
            response = await handle_chat_completion(
                request=req,
                mode="guarded",
                rails=mock_rails,
                direct_client=mock_direct_client,
            )
            responses.append(response)
        return responses

    responses = asyncio.run(run_requests())

    # Collect all response IDs
    ids = [resp.id for resp in responses]

    # Assert all IDs are unique
    assert len(set(ids)) == len(ids), (
        f"Expected {len(ids)} unique IDs but got {len(set(ids))} unique values. "
        f"Duplicate IDs found: {[id for id in ids if ids.count(id) > 1]}"
    )


# Feature: garak-endpoint-integration, Property 2: Response Schema Conformance
# **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9, 2.10**


@settings(max_examples=100, deadline=5000)
@given(request=valid_request_strategy)
def test_response_schema_conformance(request: ChatCompletionRequest):
    """Property 2: Response Schema Conformance.

    For any valid ChatCompletionRequest (containing a non-empty messages array
    with valid roles and content), the ChatCompletionResponse SHALL contain all
    required fields (id, object, created, model, choices, usage), where id is
    prefixed with "chatcmpl-", object equals "chat.completion", created is a
    positive integer, choices has exactly one element with index=0,
    message.role="assistant", and finish_reason="stop", and usage.total_tokens
    equals usage.prompt_tokens + usage.completion_tokens.
    """
    # Mock LLMRails.generate() to return {"content": "mocked response"}
    mock_rails = MagicMock()
    mock_rails.generate.return_value = {"content": "mocked response"}

    # Call handle_chat_completion in guarded mode
    response = asyncio.run(
        handle_chat_completion(
            request=request,
            mode="guarded",
            rails=mock_rails,
            direct_client=MagicMock(),
        )
    )

    # Assert id is prefixed with "chatcmpl-"
    assert response.id.startswith("chatcmpl-"), (
        f"Expected id to start with 'chatcmpl-', got '{response.id}'"
    )

    # Assert object equals "chat.completion"
    assert response.object == "chat.completion", (
        f"Expected object='chat.completion', got '{response.object}'"
    )

    # Assert created is a positive integer
    assert isinstance(response.created, int), (
        f"Expected created to be int, got {type(response.created)}"
    )
    assert response.created > 0, (
        f"Expected created > 0, got {response.created}"
    )

    # Assert model field is present and is a string
    assert isinstance(response.model, str), (
        f"Expected model to be str, got {type(response.model)}"
    )

    # Assert choices has exactly one element
    assert len(response.choices) == 1, (
        f"Expected exactly 1 choice, got {len(response.choices)}"
    )

    choice = response.choices[0]

    # Assert choice.index == 0
    assert choice.index == 0, (
        f"Expected choice.index=0, got {choice.index}"
    )

    # Assert choice.message.role == "assistant"
    assert choice.message.role == "assistant", (
        f"Expected message.role='assistant', got '{choice.message.role}'"
    )

    # Assert choice.finish_reason == "stop"
    assert choice.finish_reason == "stop", (
        f"Expected finish_reason='stop', got '{choice.finish_reason}'"
    )

    # Assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    ), (
        f"Expected total_tokens={response.usage.prompt_tokens + response.usage.completion_tokens}, "
        f"got {response.usage.total_tokens}"
    )
