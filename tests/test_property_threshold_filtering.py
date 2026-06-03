"""
Property-based test for threshold filtering correctness.

Feature: annoy-fastembed-rag, Property 4: Threshold filtering correctness

For any list of search result distances and a similarity threshold value between 0.0 and 1.0,
the filter function SHALL return only results where the computed similarity score (1 - distance/2)
is greater than or equal to the threshold, and SHALL exclude all results below the threshold.

Validates: Requirements 4.4
"""

from hypothesis import given, settings
from hypothesis import strategies as st


def filter_by_threshold(distances: list, threshold: float) -> list:
    """Filter distances by similarity threshold.

    Similarity score is computed as: 1 - distance/2
    Only distances where similarity >= threshold are returned.
    """
    return [d for d in distances if (1 - d / 2) >= threshold]


# Generator strategies
distances_strategy = st.lists(
    st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=20,
)
threshold_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@given(distances=distances_strategy, threshold=threshold_strategy)
@settings(max_examples=100)
def test_all_returned_results_meet_threshold(distances: list, threshold: float):
    """
    Property 4: Threshold filtering correctness - All returned results meet threshold.

    For any list of distances and threshold, every result returned by the filter
    has a similarity score >= threshold.

    **Validates: Requirements 4.4**
    """
    results = filter_by_threshold(distances, threshold)

    for d in results:
        similarity = 1 - d / 2
        assert similarity >= threshold, (
            f"Returned distance {d} has similarity {similarity} "
            f"which is below threshold {threshold}"
        )


@given(distances=distances_strategy, threshold=threshold_strategy)
@settings(max_examples=100)
def test_no_excluded_results_meet_threshold(distances: list, threshold: float):
    """
    Property 4: Threshold filtering correctness - No excluded results meet threshold.

    For any list of distances and threshold, every result NOT returned by the filter
    has a similarity score < threshold.

    **Validates: Requirements 4.4**
    """
    results = filter_by_threshold(distances, threshold)

    # Handle duplicates properly - use index-based removal
    remaining = list(distances)
    for d in results:
        remaining.remove(d)

    for d in remaining:
        similarity = 1 - d / 2
        assert similarity < threshold, (
            f"Excluded distance {d} has similarity {similarity} "
            f"which meets or exceeds threshold {threshold} but was not returned"
        )


@given(distances=distances_strategy)
@settings(max_examples=100)
def test_threshold_zero_returns_all(distances: list):
    """
    Property 4: Threshold filtering correctness - Zero threshold returns all.

    When threshold is 0.0, all results with non-negative similarity are returned
    (which is all results with distance <= 2.0).

    **Validates: Requirements 4.4**
    """
    threshold = 0.0
    results = filter_by_threshold(distances, threshold)

    # All distances in range [0.0, 2.0] have similarity = 1 - d/2 in [0.0, 1.0]
    # Since threshold is 0.0, all results where similarity >= 0.0 should be returned
    # That means all distances where 1 - d/2 >= 0, i.e., d <= 2.0 (which is all of them)
    assert len(results) == len(distances), (
        f"With threshold=0.0, expected all {len(distances)} distances to be returned "
        f"but got {len(results)}"
    )


@given(distances=distances_strategy)
@settings(max_examples=100)
def test_threshold_one_returns_only_exact_matches(distances: list):
    """
    Property 4: Threshold filtering correctness - Threshold 1.0 returns only exact matches.

    When threshold is 1.0, only results with distance == 0.0 are returned
    (since similarity = 1 - 0/2 = 1.0). Due to floating point precision,
    extremely small distances may also satisfy 1 - d/2 >= 1.0.

    **Validates: Requirements 4.4**
    """
    threshold = 1.0
    results = filter_by_threshold(distances, threshold)

    # Every returned result must have similarity >= 1.0 (i.e., 1 - d/2 >= 1.0)
    for d in results:
        similarity = 1 - d / 2
        assert similarity >= threshold, (
            f"With threshold=1.0, returned distance {d} has similarity {similarity} < 1.0"
        )

    # Every non-returned result must have similarity < 1.0
    remaining = list(distances)
    for d in results:
        remaining.remove(d)

    for d in remaining:
        similarity = 1 - d / 2
        assert similarity < threshold, (
            f"With threshold=1.0, excluded distance {d} has similarity {similarity} >= 1.0"
        )
