"""A candidate chunk is accepted only if its deadline is still valid."""

import numpy as np
import pytest
from candidate_chunks import classify_candidate_chunks


def test_chunks_preserve_all_ordered_verdicts_and_duplicates():
    rows = np.array([[1.0], [2.0], [1.0], [3.0]])
    batches = []

    def check(part):
        batches.append(part.copy())
        return ["clear" if row[0] == 1 else "sampled_collision" for row in part]

    assert classify_candidate_chunks(rows, check, lambda: None, chunk_size=3) == [
        "clear",
        "sampled_collision",
        "clear",
        "sampled_collision",
    ]
    assert [len(part) for part in batches] == [3, 1]


def test_expired_chunk_never_returns_partial_or_complete_results():
    calls = 0

    def budget():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise TimeoutError("budget exhausted")

    with pytest.raises(TimeoutError, match="budget exhausted"):
        classify_candidate_chunks(
            np.ones((3, 1)), lambda part: ["clear"] * len(part), budget, chunk_size=2
        )
