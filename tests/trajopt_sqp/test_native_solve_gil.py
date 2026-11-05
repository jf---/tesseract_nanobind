"""Native solve entry points allow unrelated Python threads to make progress."""

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import numpy as np
import pytest
import scipy.sparse

from tesseract_robotics import trajopt_ifopt as ti
from tesseract_robotics import trajopt_sqp as tsqp

# Long enough that normal bytecode switching cannot mask a held GIL.
LONG_SWITCH_INTERVAL_S = 10.0
# A sparse convex problem large enough for the native operation to be observable.
QP_DIMENSION = 2000


def _assert_releases_gil(operation):
    started = Event()
    state = {"running": False, "error": None}

    def worker():
        state["running"] = True
        started.set()
        try:
            operation()
        except BaseException as exc:
            state["error"] = exc
        finally:
            state["running"] = False

    previous = sys.getswitchinterval()
    sys.setswitchinterval(LONG_SWITCH_INTERVAL_S)
    try:
        thread = Thread(target=worker)
        thread.start()
        started.wait()
        observed = state["running"]
        thread.join()
    finally:
        sys.setswitchinterval(previous)
    if state["error"] is not None:
        raise state["error"]
    assert observed, "Python thread did not run during native solve"


def _qp_solver(solver_type):
    solver = solver_type()
    assert solver.init(QP_DIMENSION, QP_DIMENSION)
    identity = scipy.sparse.identity(QP_DIMENSION, format="csc")
    assert solver.updateHessianMatrix(identity)
    assert solver.updateGradient(-np.ones(QP_DIMENSION))
    assert solver.updateLinearConstraintsMatrix(identity)
    assert solver.updateBounds(-np.ones(QP_DIMENSION), np.ones(QP_DIMENSION))
    return solver


@pytest.mark.parametrize("solver_type", [tsqp.PIQPSolver, tsqp.OSQPEigenSolver])
def test_qp_solve_releases_gil(solver_type):
    solver = _qp_solver(solver_type)
    _assert_releases_gil(solver.solve)


@pytest.mark.parametrize("solver_type", [tsqp.PIQPSolver, tsqp.OSQPEigenSolver])
def test_independent_qp_solvers_run_in_threads(solver_type):
    first, second = _qp_solver(solver_type), _qp_solver(solver_type)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first.solve)
        second_result = pool.submit(second.solve)
        assert first_result.result()
        assert second_result.result()
    np.testing.assert_allclose(first.getSolution(), second.getSolution())


def _problem_and_solver(callback):
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
    solver = tsqp.TrustRegionSQPSolver(tsqp.PIQPSolver())
    solver.registerCallback(callback)
    solver.init(problem)
    return problem, solver


def test_step_calls_python_callback():
    class Callback(tsqp.SQPCallback):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def execute(self, _problem, _results):
            self.calls += 1
            return True

    callback = Callback()
    problem, solver = _problem_and_solver(callback)
    solver.stepSQPSolver()
    assert problem is not None
    assert callback.calls > 0


def test_step_propagates_python_callback_exception():
    class Callback(tsqp.SQPCallback):
        def execute(self, _problem, _results):
            raise RuntimeError("callback failed")

    callback = Callback()
    problem, solver = _problem_and_solver(callback)
    with pytest.raises(RuntimeError, match="callback failed"):
        solver.stepSQPSolver()
    assert problem is not None


def test_sqp_step_releases_gil():
    joint_names = [f"j{i}" for i in range(QP_DIMENSION)]
    nodes = ti.createNodesVariables(
        "trajectory",
        joint_names,
        [np.zeros(QP_DIMENSION), np.full(QP_DIMENSION, 0.5)],
        ti.toBounds(-np.ones(QP_DIMENSION), np.ones(QP_DIMENSION)),
    )
    variables = [node.getVar("joints") for node in nodes.getNodes()]
    problem = tsqp.TrajOptQPProblem(nodes)
    problem.addCostSet(
        ti.JointVelConstraint(np.zeros(QP_DIMENSION), variables, np.ones(QP_DIMENSION), "Velocity"),
        tsqp.CostPenaltyType.SQUARED,
    )
    problem.setup()
    solver = tsqp.TrustRegionSQPSolver(tsqp.PIQPSolver())
    solver.init(problem)
    _assert_releases_gil(solver.stepSQPSolver)


def test_step_propagates_python_constraint_exception():
    class Cost(ti.ConstraintSet):
        def __init__(self):
            super().__init__("python_cost", 1)
            self.fail = False
            self.calls = 0

        def getValues(self):
            self.calls += 1
            if self.fail:
                raise IndexError("Python cost failure")
            return np.array([self.getVariables().getValues()[0] - 0.25])

        def getBounds(self):
            return [ti.Bounds(0.0, 0.0)]

        def getJacobian(self):
            count = len(self.getVariables().getValues())
            return scipy.sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, count))

        def update(self):
            return self.getRows()

        def getCoefficients(self):
            return np.ones(1)

    nodes = ti.createNodesVariables(
        "trajectory",
        ["joint"],
        [np.array([0.5])],
        ti.toBounds(-np.ones(1), np.ones(1)),
    )
    cost = Cost()
    problem = tsqp.TrajOptQPProblem(nodes)
    problem.addCostSet(cost, tsqp.CostPenaltyType.SQUARED)
    problem.setup()
    solver = tsqp.TrustRegionSQPSolver(tsqp.PIQPSolver())
    solver.init(problem)
    assert cost.calls > 0
    cost.fail = True
    with pytest.raises(IndexError, match="Python cost failure"):
        solver.stepSQPSolver()


@pytest.mark.parametrize("method", ["runTrustRegionLoop", "solveQPProblem"])
def test_incremental_methods_propagate_callback_exception(method):
    class Callback(tsqp.SQPCallback):
        def __init__(self):
            super().__init__()
            self.fail = False
            self.calls = 0

        def execute(self, _problem, _results):
            self.calls += 1
            if self.fail:
                raise RuntimeError("incremental callback failure")
            return True

    callback = Callback()
    problem, solver = _problem_and_solver(callback)
    solver.stepSQPSolver()
    assert callback.calls > 0
    callback.fail = True
    with pytest.raises(RuntimeError, match="incremental callback failure"):
        getattr(solver, method)()
    assert problem is not None
