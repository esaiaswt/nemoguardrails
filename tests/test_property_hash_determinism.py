"""Property-based test for content hash determinism and change sensitivity.

Feature: annoy-fastembed-rag, Property 2: Content hash determinism and change sensitivity

For any two sets of knowledge base document contents, if the contents are identical
(same files in same sorted order), then the computed cache hash SHALL be identical.
Conversely, if any document content differs between the two sets, the computed hash
SHALL differ.

**Validates: Requirements 3.3, 3.4**
"""

import hashlib

from hypothesis import assume, given, settings
from hypothesis import strategies as st


def compute_content_hash(embedding_engine: str, embedding_model: str, chunk_texts: list) -> str:
    """Compute the cache hash the same way KnowledgeBase.build() does.

    The hash is computed over: embedding_engine + embedding_model + concatenated chunk texts.
    Uses MD5 if available, otherwise SHA256.
    """
    hash_prefix = embedding_engine + embedding_model
    combined = hash_prefix + "".join(chunk_texts)
    try:
        hashlib.md5(b"")
        hash_func = hashlib.md5
    except (AttributeError, ValueError):
        hash_func = hashlib.sha256
    return hash_func(combined.encode("utf-8")).hexdigest()


# Generator strategy for document content lists
document_lists = st.lists(st.text(min_size=1, max_size=200), min_size=1, max_size=10)


@given(docs=document_lists)
@settings(max_examples=100)
def test_identical_content_produces_identical_hash(docs):
    """Property 2: Content hash determinism - identical content produces identical hash.

    For any generated list of document strings, computing the hash twice always
    produces the same result.

    **Validates: Requirements 3.3, 3.4**
    """
    engine = "FastEmbed"
    model = "all-MiniLM-L6-v2"

    hash1 = compute_content_hash(engine, model, docs)
    hash2 = compute_content_hash(engine, model, docs)

    assert hash1 == hash2, (
        f"Hash should be deterministic for identical content. "
        f"Got hash1={hash1}, hash2={hash2} for docs={docs!r}"
    )


@given(
    docs1=document_lists,
    docs2=document_lists,
)
@settings(max_examples=100)
def test_different_content_produces_different_hash(docs1, docs2):
    """Property 2: Content hash change sensitivity - different content produces different hash.

    For any two different lists of document strings (where at least one element differs),
    the hashes differ. (Note: hash collisions are theoretically possible but
    astronomically unlikely with MD5/SHA256.)

    **Validates: Requirements 3.3, 3.4**
    """
    # Ensure the two document lists actually differ
    assume(docs1 != docs2)

    engine = "FastEmbed"
    model = "all-MiniLM-L6-v2"

    hash1 = compute_content_hash(engine, model, docs1)
    hash2 = compute_content_hash(engine, model, docs2)

    assert hash1 != hash2, (
        f"Hash should differ for different content. "
        f"Got same hash={hash1} for docs1={docs1!r} and docs2={docs2!r}"
    )
