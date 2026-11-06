"""Opt-in PIQP timing records distinguish setup, update and rejected calls."""

import numpy as np
import scipy.sparse

from tesseract_robotics import trajopt_sqp as sqp

# Absolute joint tolerance covers PIQP default 1e-4 convergence tolerance.
SOLUTION_ATOL = 1e-3


def _configured_solver():
    solver = sqp.PIQPTimedSolver()
    assert solver.settings.compute_timings
    assert solver.init(2, 2)
    assert solver.updateHessianMatrix(scipy.sparse.identity(2, format="csc"))
    assert solver.updateGradient(-np.ones(2))
    assert solver.updateLinearConstraintsMatrix(scipy.sparse.identity(2, format="csc"))
    assert solver.updateBounds(np.zeros(2), np.ones(2))
    return solver


def test_setup_and_bounds_update_are_recorded_without_changing_solutions():
    solver = _configured_solver()
    assert solver.solve()
    first = solver.getTimingRecords()[0]
    assert first["success"] is True
    assert first["mode"] == "setup"
    assert first["native_status"] == "solved"
    assert first["setup_s"] >= 0
    assert first["update_s"] is None
    assert first["solve_s"] >= 0
    assert first["wall_s"] >= first["solve_s"]

    assert solver.updateBounds(np.zeros(2), np.full(2, 0.5))
    assert solver.solve()
    second = solver.getTimingRecords()[1]
    assert second["success"] is True
    assert second["mode"] == "update"
    assert second["setup_s"] is None
    assert second["update_s"] >= 0
    assert second["solve_s"] >= 0
    np.testing.assert_allclose(solver.getSolution(), [0.5, 0.5], atol=SOLUTION_ATOL)

    assert solver.updateGradient(np.zeros(2))
    assert solver.solve()
    assert solver.getTimingRecords()[2]["mode"] == "setup"


def test_early_failure_never_reports_stale_native_timing():
    solver = _configured_solver()
    assert solver.solve()
    crossed = scipy.sparse.csc_matrix([[1.0, 0.0], [1.0, 0.0]])
    assert solver.updateLinearConstraintsMatrix(crossed)
    assert solver.updateBounds(np.array([0.0, 2.0]), np.array([1.0, 3.0]))
    assert not solver.solve()
    record = solver.getTimingRecords()[-1]
    assert record["success"] is False
    assert record["mode"] == "unknown"
    assert record["native_status"] is None
    assert record["setup_s"] is None
    assert record["update_s"] is None
    assert record["solve_s"] is None
    assert record["wall_s"] >= 0
    solver.clearTimingRecords()
    assert solver.getTimingRecords() == []


def test_timed_problem_records_convexification_and_exact_evaluations():
    from tesseract_robotics import trajopt_ifopt as ti

    def make_problem(problem_type):
        nodes = ti.createNodesVariables(
            "trajectory",
            ["joint"],
            [np.array([0.0]), np.array([0.5]), np.array([1.0])],
            ti.toBounds(-np.ones(1), np.ones(1)),
        )
        variables = [node.getVar("joints") for node in nodes.getNodes()]
        problem = problem_type(nodes)
        velocity = ti.JointVelConstraint(np.zeros(1), variables, np.ones(1), "velocity")
        problem.addCostSet(velocity, sqp.CostPenaltyType.SQUARED)
        problem.setup()
        return nodes, problem

    original_nodes, original = make_problem(sqp.TrajOptQPProblem)
    timed_nodes, timed = make_problem(sqp.TimedTrajOptQPProblem)
    original_solver = sqp.TrustRegionSQPSolver(sqp.PIQPSolver())
    timed_solver = sqp.TrustRegionSQPSolver(sqp.PIQPTimedSolver())
    original_solver.solve(original)
    timed_solver.solve(timed)
    assert original_solver.getStatus() == timed_solver.getStatus()
    np.testing.assert_allclose(
        timed_nodes.getValues(), original_nodes.getValues(), atol=SOLUTION_ATOL
    )
    timings = timed.getEvaluationTimings()
    for name in (
        "set_variables",
        "convexify",
        "exact_costs",
        "exact_constraints",
        "convex_costs",
        "convex_constraints",
    ):
        assert timings[name]["calls"] > 0
        assert timings[name]["seconds"] >= 0
