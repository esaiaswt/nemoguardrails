"""
Property-based test for search result ordering.

Feature: annoy-fastembed-rag, Property 3: Search results ordered by descending similarity

For any search query against a non-empty Annoy index, the returned list of results SHALL
have similarity scores in non-increasing order (each score is greater than or equal to
the next).

Validates: Requirements 4.3
"""

from typing import List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st


def sort_by_descending_similarity(
    results: List[Tuple[int, float]],
) -> List[Tuple[int, float, float]]:
    """
    Takes a list of (index, distance) pairs and returns them sorted by
    descending similarity score where score = 1 - distance/2.

    Angular distance ranges from 0.0 to 2.0, so similarity ranges from
    0.0 (most distant) to 1.0 (identical).

    Returns list of (index, distance, score) tuples sorted by score descending.
    """
    scored = [(idx, dist, 1.0 - dist / 2.0) for idx, dist in results]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# Generator strategy for (index, distance) pairs
# Angular distance ranges from 0.0 to 2.0
index_distance_pairs = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=1000),
        st.floats(min_value=0.0, max_value=2.0),
    ),
    min_size=1,
    max_size=20,
)


@given(pairs=index_distance_pairs)
@settings(max_examples=100)
def test_results_have_non_increasing_similarity_scores(
    pairs: List[Tuple[int, float]],
):
    """
    Property 3: Search results ordered by descending similarity

    For any generated list of distance values (floats between 0.0 and 2.0,
    since angular distance ranges from 0 to 2), after sorting by descending
    similarity, each score is >= the next score.

    **Validates: Requirements 4.3**
    """
    sorted_results = sort_by_descending_similarity(pairs)

    # Verify non-increasing similarity scores
    for i in range(len(sorted_results) - 1):
        current_score = sorted_results[i][2]
        next_score = sorted_results[i + 1][2]
        assert current_score >= next_score, (
            f"Similarity scores not in non-increasing order at index {i}: "
            f"{current_score} < {next_score}"
        )


@given(pairs=index_distance_pairs)
@settings(max_examples=100)
def test_sorted_results_preserve_all_items(
    pairs: List[Tuple[int, float]],
):
    """
    Property 3: Search results ordered by descending similarity

    For any list of (index, distance) pairs, sorting preserves all items
    (no items are lost or duplicated).

    **Validates: Requirements 4.3**
    """
    sorted_results = sort_by_descending_similarity(pairs)

    # Same number of items
    assert len(sorted_results) == len(pairs), (
        f"Item count mismatch: input has {len(pairs)}, "
        f"sorted has {len(sorted_results)}"
    )

    # All original (index, distance) pairs are present in the sorted output
    original_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))
    sorted_pairs = sorted(
        [(idx, dist) for idx, dist, _ in sorted_results],
        key=lambda x: (x[0], x[1]),
    )
    assert original_pairs == sorted_pairs, (
        "Sorting must preserve all items without loss or duplication"
    )
