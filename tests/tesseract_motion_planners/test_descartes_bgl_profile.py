"""The optional BGL profile preserves the default ladder and thread guard."""

import numpy as np
import pytest

from tesseract_robotics import tesseract_motion_planners_descartes as descartes


def test_bgl_profile_is_explicit_and_single_threaded():
    profile = descartes.DescartesBGLDijkstraSolverProfileD()
    assert profile.num_threads == 1
    with pytest.raises(descartes.BGLThreadCountError, match="num_threads must be 1"):
        profile.num_threads = 2
    assert descartes.DescartesLadderGraphSolverProfileD().num_threads == 1


def test_scaled_squared_distance_and_invalid_scale():
    with pytest.raises(descartes.EdgeScaleError):
        descartes.ScaledSquaredL2EdgeEvaluatorD(np.array([0.0]))
    evaluator = descartes.ScaledSquaredL2EdgeEvaluatorD(np.array([2.0, 0.5]))
    start = descartes.DescartesStateD(np.array([0.0, 0.0]))
    end = descartes.DescartesStateD(np.array([1.0, -1.0]))
    valid, cost = evaluator.evaluate(start, end)
    assert valid
    assert cost == pytest.approx(4.25)
    with pytest.raises(descartes.EdgeDofMismatchError):
        evaluator.evaluate(start, descartes.DescartesStateD(np.array([1.0])))
    with pytest.raises(descartes.EdgeCostError):
        evaluator.evaluate(start, descartes.DescartesStateD(np.array([1.0, np.nan])))
