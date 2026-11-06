"""Type-check a consumer of both native candidate classifiers."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tesseract_robotics.tesseract_collision import (
    DiscreteContactManager,
    classify_candidate_state_rows,
    classify_candidate_state_rows_first,
)
from tesseract_robotics.tesseract_state_solver import StateSolver


def classify(
    solver: StateSolver,
    manager: DiscreteContactManager,
    names: Sequence[str],
    rows: NDArray[np.float64],
    limits: NDArray[np.float64],
) -> tuple[list[str], list[str]]:
    return (
        classify_candidate_state_rows(solver, manager, names, rows, limits),
        classify_candidate_state_rows_first(solver, manager, names, rows, limits),
    )
