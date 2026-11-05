"""PIQP's Python boundary validates every QP dimension before native calls."""

import numpy as np
import pytest
import scipy.sparse

from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as tsqp


def test_modern_sqp_iteration_parameter():
    parameters = tsqp.SQPParameters()
    parameters.max_iter = 7
    assert parameters.max_iter == 7
    assert not hasattr(parameters, "max_iterations")


def test_negative_prediction_status_is_distinct():
    status = tsqp.SQPStatus.INVALID_PREDICTED_IMPROVEMENT
    assert status != tsqp.SQPStatus.NLP_CONVERGED
    assert tsqp.toString(status) == "SQP_INVALID_PREDICTED_IMPROVEMENT"


def test_constraint_violation_reduction_and_solver_init():
    nodes = ti.createNodesVariables(
        "trajectory",
        ["j0", "j1"],
        [np.array([0.25, 0.5])],
        ti.toBounds(-np.ones(2), np.ones(2)),
    )
    problem = tsqp.TrajOptQPProblem(nodes)
    problem.addConstraintSet(
        ti.JointPosConstraint(np.zeros(2), nodes.getNodes()[0].getVar("joints"), np.ones(2), "pin")
    )
    problem.setup()
    problem.setSumConstraintSetViolations(True)
    np.testing.assert_allclose(problem.getExactConstraintViolations().raw, [0.75])
    problem.setSumConstraintSetViolations(False)
    np.testing.assert_allclose(problem.getExactConstraintViolations().raw, [0.5])

    solver = tsqp.TrustRegionSQPSolver(tsqp.PIQPSolver())
    solver.params.sum_constraint_set_violations = True
    solver.init(problem)
    np.testing.assert_allclose(problem.getExactConstraintViolations().raw, [0.75])
    solver.params.sum_constraint_set_violations = False
    solver.init(problem)
    np.testing.assert_allclose(problem.getExactConstraintViolations().raw, [0.5])


def test_piqp_checked_updates_and_reinitialization():
    solver = tsqp.PIQPSolver()
    assert solver.init(2, 3)
    invalid_updates = (
        (solver.updateHessianMatrix, (scipy.sparse.csr_matrix((1, 2)),)),
        (solver.updateHessianMatrix, (scipy.sparse.csr_matrix((2, 1)),)),
        (solver.updateGradient, (np.zeros(1),)),
        (solver.updateLowerBound, (np.zeros(2),)),
        (solver.updateUpperBound, (np.zeros(2),)),
        (solver.updateBounds, (np.zeros(2), np.zeros(3))),
        (solver.updateBounds, (np.zeros(3), np.zeros(2))),
        (solver.updateLinearConstraintsMatrix, (scipy.sparse.csr_matrix((2, 2)),)),
        (solver.updateLinearConstraintsMatrix, (scipy.sparse.csr_matrix((3, 1)),)),
    )
    for update, arguments in invalid_updates:
        with pytest.raises(tsqp.QPSizeError):
            update(*arguments)
    assert solver.updateHessianMatrix(scipy.sparse.identity(2, format="csr"))
    assert solver.updateGradient(np.zeros(2))
    assert solver.updateLinearConstraintsMatrix(scipy.sparse.csr_matrix((3, 2)))
    assert solver.updateBounds(np.zeros(3), np.ones(3))
    assert solver.init(4, 1)
    with pytest.raises(tsqp.QPSizeError):
        solver.updateGradient(np.zeros(2))
    assert solver.updateGradient(np.zeros(4))
    assert solver.clear()
    with pytest.raises(tsqp.QPSizeError):
        solver.updateGradient(np.zeros(4))


def test_piqp_sqp_callback_keeps_problem_live():
    nodes = ti.createNodesVariables(
        "trajectory",
        ["joint"],
        [np.array([0.0]), np.array([0.5])],
        ti.toBounds(-np.ones(1), np.ones(1)),
    )
    variables = [node.getVar("joints") for node in nodes.getNodes()]
    problem = tsqp.TrajOptQPProblem(nodes)
    problem.addCostSet(
        ti.JointVelConstraint(np.zeros(1), variables, np.ones(1), "Velocity"),
        tsqp.CostPenaltyType.SQUARED,
    )
    problem.setup()

    class Callback(tsqp.SQPCallback):
        def __init__(self):
            super().__init__()
            self.problems = []

        def execute(self, current_problem, _results):
            self.problems.append(current_problem)
            return True

    callback = Callback()
    solver = tsqp.TrustRegionSQPSolver(tsqp.PIQPSolver())
    solver.registerCallback(callback)
    solver.solve(problem)
    assert callback.problems
    assert all(item is problem for item in callback.problems)
