"""Property-based test for chunk formatting includes heading label and body content.

Feature: annoy-fastembed-rag, Property 5: Chunk formatting includes heading label and body content

For any document chunk with a non-empty title and non-empty body, the formatted
chunk text SHALL contain the title as a heading label and SHALL contain the body content.

**Validates: Requirements 6.2**
"""

from hypothesis import given, settings
from hypothesis import strategies as st


def format_chunk(title: str, body: str) -> str:
    """Format a document chunk by prepending its source heading as a label.
    Returns the formatted text with title as heading and body content."""
    return f"# {title}\n\n{body}"


# Generator strategies
titles = st.text(
    min_size=1,
    max_size=100,
    alphabet=st.characters(whitelist_categories=('L', 'N', 'Z', 'P')),
)
bodies = st.text(min_size=1, max_size=500)


@given(title=titles, body=bodies)
@settings(max_examples=100)
def test_formatted_chunk_contains_title(title, body):
    """Property 5: Formatted chunk text contains the title.

    For any non-empty title and non-empty body, the formatted chunk text
    contains the title.

    **Validates: Requirements 6.2**
    """
    result = format_chunk(title, body)

    assert title in result, (
        f"Formatted chunk should contain the title. "
        f"Title={title!r} not found in result={result!r}"
    )


@given(title=titles, body=bodies)
@settings(max_examples=100)
def test_formatted_chunk_contains_body(title, body):
    """Property 5: Formatted chunk text contains the body content.

    For any non-empty title and non-empty body, the formatted chunk text
    contains the body content.

    **Validates: Requirements 6.2**
    """
    result = format_chunk(title, body)

    assert body in result, (
        f"Formatted chunk should contain the body content. "
        f"Body={body!r} not found in result={result!r}"
    )


@given(title=titles, body=bodies)
@settings(max_examples=100)
def test_formatted_chunk_has_heading_prefix(title, body):
    """Property 5: Formatted text starts with # followed by the title.

    For any non-empty title, the formatted text starts with `#` followed
    by the title.

    **Validates: Requirements 6.2**
    """
    result = format_chunk(title, body)

    expected_prefix = f"# {title}"
    assert result.startswith(expected_prefix), (
        f"Formatted chunk should start with '# {{title}}'. "
        f"Expected prefix={expected_prefix!r}, got start={result[:len(expected_prefix)+10]!r}"
    )
