"""Opt-in deadline-bounded candidate classification."""

import numpy as np


class CandidateChunkError(RuntimeError):
    """Invalid candidate chunk request or native result."""


def classify_candidate_chunks(rows, check_rows, budget, *, chunk_size):
    """Classify ordered rows, discarding all work if a chunk exceeds the budget.

    The caller owns the checker and applies its ACM and margins. The deadline is
    checked before and after each chunk, not between individual native checks.
    """
    array = np.asarray(rows, dtype=float)
    if array.ndim != 2 or len(array) == 0 or not np.isfinite(array).all():
        raise CandidateChunkError("finite nonempty candidate matrix required")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise CandidateChunkError("positive integer chunk size required")
    verdicts = []
    for start in range(0, len(array), chunk_size):
        part = array[start : start + chunk_size]
        budget()
        batch = check_rows(part)
        budget()
        if len(batch) != len(part) or any(
            x not in ("clear", "joint_limit", "sampled_collision") for x in batch
        ):
            raise CandidateChunkError("native classifier returned an invalid verdict sequence")
        verdicts.extend(batch)
    return verdicts
