"""Property-based test for markdown chunking at heading boundaries.

Feature: annoy-fastembed-rag, Property 1: Markdown chunking splits at heading boundaries

For any valid markdown document containing one or more headings (# through ######),
splitting the document into topic chunks SHALL produce chunks where each chunk's body
contains no markdown heading lines (lines starting with #), and the total content
across all chunks accounts for all non-heading text in the original document.

**Validates: Requirements 1.2, 1.3**
"""

from hypothesis import given, settings
from hypothesis import strategies as st


def split_markdown_into_chunks(markdown_text: str) -> list[dict]:
    """Split markdown into chunks at heading boundaries.
    Returns list of {"title": str, "body": str} dicts."""
    chunks = []
    current_title = ""
    current_body_lines = []

    for line in markdown_text.split("\n"):
        if line.startswith("#"):
            # Save previous chunk
            if current_title or current_body_lines:
                chunks.append({"title": current_title, "body": "\n".join(current_body_lines)})
            current_title = line.lstrip("#").strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)

    # Save last chunk
    if current_title or current_body_lines:
        chunks.append({"title": current_title, "body": "\n".join(current_body_lines)})

    return chunks


@st.composite
def markdown_with_headings(draw):
    """Generate random markdown documents with headings at various levels and varying body lengths."""
    num_sections = draw(st.integers(min_value=1, max_value=5))
    lines = []
    for _ in range(num_sections):
        level = draw(st.integers(min_value=1, max_value=6))
        title = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
                min_size=1,
                max_size=30,
            )
        )
        heading = "#" * level + " " + title
        lines.append(heading)
        body_lines_count = draw(st.integers(min_value=0, max_value=5))
        for _ in range(body_lines_count):
            # Body text must not start with #
            body = draw(st.text(min_size=0, max_size=50).filter(lambda x: not x.startswith("#")))
            lines.append(body)
    return "\n".join(lines)


@st.composite
def markdown_without_headings(draw):
    """Generate random markdown text that contains no heading lines (no lines starting with #)."""
    num_lines = draw(st.integers(min_value=1, max_value=10))
    lines = []
    for _ in range(num_lines):
        line = draw(st.text(min_size=0, max_size=50).filter(lambda x: not x.startswith("#")))
        lines.append(line)
    return "\n".join(lines)


@given(md=markdown_with_headings())
@settings(max_examples=100)
def test_chunk_bodies_contain_no_heading_lines(md):
    """Property 1: After chunking, no chunk body contains a line starting with #.

    For any markdown document with headings, splitting at heading boundaries produces
    chunks whose body text never contains heading lines.

    **Validates: Requirements 1.2, 1.3**
    """
    chunks = split_markdown_into_chunks(md)

    for chunk in chunks:
        body_lines = chunk["body"].split("\n")
        for line in body_lines:
            assert not line.startswith("#"), (
                f"Chunk body should not contain heading lines. "
                f"Found heading line '{line}' in chunk with title '{chunk['title']}'"
            )


@given(md=markdown_with_headings())
@settings(max_examples=100)
def test_total_content_accounts_for_all_non_heading_text(md):
    """Property 1: The combined body text from all chunks contains all non-heading lines from the original.

    For any markdown document, after chunking, the concatenated body content accounts
    for all non-heading lines from the original document (preserving content, preserving order).

    **Validates: Requirements 1.2, 1.3**
    """
    chunks = split_markdown_into_chunks(md)

    # Collect all non-heading lines from the original document
    original_non_heading_lines = [
        line for line in md.split("\n") if not line.startswith("#")
    ]

    # Collect body lines from chunks. The chunking function builds body as
    # "\n".join(body_lines). When body_lines is [], body becomes "".
    # When body_lines is [""], body is also "". To avoid this ambiguity in
    # round-tripping, we count how many body lines each chunk should have
    # by tracking through the original lines directly.
    #
    # Simpler approach: verify all non-empty non-heading lines are preserved
    # in order, and that total non-empty content matches. Empty lines between
    # consecutive headings are an edge case where the representation is lossy.
    all_chunk_body_lines = []
    for chunk in chunks:
        if chunk["body"]:
            all_chunk_body_lines.extend(chunk["body"].split("\n"))

    # Verify all non-empty non-heading lines are preserved in order
    original_non_empty = [line for line in original_non_heading_lines if line]
    chunks_non_empty = [line for line in all_chunk_body_lines if line]

    assert chunks_non_empty == original_non_empty, (
        f"All non-empty non-heading lines should be preserved in chunk bodies in order. "
        f"Original: {original_non_empty!r}, From chunks: {chunks_non_empty!r}"
    )


@given(md=markdown_without_headings())
@settings(max_examples=100)
def test_document_without_headings_is_single_chunk(md):
    """Property 1: For any markdown text with no heading lines, chunking produces exactly one chunk.

    When a document contains no headings, the entire content is treated as a single
    chunk with an empty title.

    **Validates: Requirements 1.2, 1.3**
    """
    chunks = split_markdown_into_chunks(md)

    assert len(chunks) == 1, (
        f"Document without headings should produce exactly one chunk. "
        f"Got {len(chunks)} chunks for text: {md!r}"
    )
    assert chunks[0]["title"] == "", (
        f"Single chunk from headingless document should have empty title. "
        f"Got title: {chunks[0]['title']!r}"
    )
    assert chunks[0]["body"] == md, (
        f"Single chunk body should equal the entire document. "
        f"Got body: {chunks[0]['body']!r}, expected: {md!r}"
    )
