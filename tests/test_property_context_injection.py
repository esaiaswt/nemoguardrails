"""
Property-based test for context injection character limit.

Feature: annoy-fastembed-rag, Property 6: Context injection respects character limit

For any set of ranked relevant chunks where the combined text exceeds 2000 characters,
the injected context SHALL not exceed 2000 characters, and higher-ranked chunks
(by similarity score) SHALL be preserved over lower-ranked chunks.

Validates: Requirements 6.4
"""

from hypothesis import given, settings
from hypothesis import strategies as st


def inject_context(chunks: list[dict], max_chars: int = 2000) -> str:
    """Inject ranked chunks into context string, respecting character limit.

    Chunks are assumed to be in descending similarity order (highest first).
    Higher-ranked chunks are preserved over lower-ranked ones.
    Chunks are separated by blank lines.

    Args:
        chunks: List of {"title": str, "body": str, "score": float} dicts,
                ordered by descending score.
        max_chars: Maximum character limit for the combined context.

    Returns:
        Formatted context string within the character limit.
    """
    context_parts = []
    total_chars = 0
    separator = "\n\n"

    for chunk in chunks:
        formatted = f"# {chunk['title']}\n\n{chunk['body']}"
        added_length = len(formatted) + (len(separator) if context_parts else 0)

        if total_chars + added_length > max_chars:
            break

        context_parts.append(formatted)
        total_chars += added_length

    return separator.join(context_parts)


# Generator strategies
@st.composite
def ranked_chunks(draw):
    """Generate a list of chunks sorted by descending score (as from retrieval)."""
    num_chunks = draw(st.integers(min_value=1, max_value=10))
    chunks = []
    for i in range(num_chunks):
        title = draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
            )
        )
        body = draw(st.text(min_size=1, max_size=500))
        score = draw(st.floats(min_value=0.0, max_value=1.0))
        chunks.append({"title": title, "body": body, "score": score})
    # Sort by descending score (as they would be from retrieval)
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


@given(chunks=ranked_chunks())
@settings(max_examples=100)
def test_context_does_not_exceed_character_limit(chunks: list[dict]):
    """
    Property 6: Context injection respects character limit - output within limit.

    For any list of chunks, the injected context length does not exceed 2000 characters.

    **Validates: Requirements 6.4**
    """
    result = inject_context(chunks)

    assert len(result) <= 2000, (
        f"Injected context length {len(result)} exceeds 2000 character limit. "
        f"Number of chunks: {len(chunks)}"
    )


@given(chunks=ranked_chunks())
@settings(max_examples=100)
def test_higher_ranked_chunks_preserved(chunks: list[dict]):
    """
    Property 6: Context injection respects character limit - higher-ranked preserved.

    For any list of chunks where the total exceeds 2000 chars, all chunks included
    in the output appear before any excluded chunks in the input order
    (higher-ranked chunks are kept).

    **Validates: Requirements 6.4**
    """
    result = inject_context(chunks)

    # Determine which chunks are included vs excluded
    included_indices = []
    excluded_indices = []

    for i, chunk in enumerate(chunks):
        formatted = f"# {chunk['title']}\n\n{chunk['body']}"
        if formatted in result:
            included_indices.append(i)
        else:
            excluded_indices.append(i)

    # All included indices should come before all excluded indices
    # (since chunks are in descending score order, included ones should be
    # from the beginning of the list)
    if included_indices and excluded_indices:
        max_included = max(included_indices)
        min_excluded = min(excluded_indices)
        assert max_included < min_excluded, (
            f"Higher-ranked chunk at index {min_excluded} was excluded while "
            f"lower-ranked chunk at index {max_included} was included. "
            f"Included indices: {included_indices}, Excluded indices: {excluded_indices}"
        )


@given(data=st.data())
@settings(max_examples=100)
def test_empty_chunks_produce_empty_context(data):
    """
    Property 6: Context injection respects character limit - empty input.

    When chunks list is empty, context is empty string.

    **Validates: Requirements 6.4**
    """
    result = inject_context([])

    assert result == "", (
        f"Expected empty string for empty chunks list, got: '{result}'"
    )


@given(
    title=st.text(
        min_size=1,
        max_size=50,
        alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
    ),
    body=st.text(min_size=1, max_size=500),
    score=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_single_chunk_under_limit_is_included(title: str, body: str, score: float):
    """
    Property 6: Context injection respects character limit - single chunk inclusion.

    When a single chunk is under the limit, it's fully included.

    **Validates: Requirements 6.4**
    """
    chunk = {"title": title, "body": body, "score": score}
    formatted = f"# {title}\n\n{body}"

    # Only test when the single chunk is under the limit
    if len(formatted) <= 2000:
        result = inject_context([chunk])
        assert result == formatted, (
            f"Single chunk under limit was not fully included. "
            f"Expected: '{formatted}', Got: '{result}'"
        )
