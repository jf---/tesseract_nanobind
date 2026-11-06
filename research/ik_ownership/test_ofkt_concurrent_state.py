"""The OFKT-only state entry point accepts owned inputs and permits thread progress."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from tesseract_robotics.tesseract_state_solver import KDLStateSolver, OFKTStateSolver
from tests.test_gil_entry_points import assert_releases_gil

pytest_plugins = ["tests.tesseract_time_parameterization.test_time_parameterization"]


def test_ofkt_concurrent_state_releases_gil(abb_env):
    solver = abb_env.getStateSolver()
    assert isinstance(solver, OFKTStateSolver)
    joints = {str(name): 0.1 for name in solver.getActiveJointIds()}
    expected = solver.getState(joints)
    assert_releases_gil(lambda: solver.getStateConcurrent(joints))
    assert solver.getStateConcurrent(joints).joints == expected.joints


def test_ofkt_state_allows_concurrent_mutation(abb_env):
    solver = abb_env.getStateSolver()
    assert isinstance(solver, OFKTStateSolver)
    names = [str(name) for name in solver.getActiveJointIds()]
    wanted = {name: 0.1 for name in names}
    alternate = {name: 0.2 for name in names}

    def read():
        for _ in range(100):
            state = solver.getStateConcurrent(wanted)
            assert all(state.joints[name] == value for name, value in wanted.items())

    def mutate():
        for _ in range(100):
            solver.setState(alternate)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(read), pool.submit(mutate)]
        for future in futures:
            future.result()


def test_ofkt_concurrent_state_rejects_unknown_joint(abb_env):
    solver = abb_env.getStateSolver()
    assert isinstance(solver, OFKTStateSolver)
    with pytest.raises(ValueError, match="unknown or non-active"):
        solver.getStateConcurrent({"nonexistent_joint": 0.0})


def test_kdl_does_not_expose_ofkt_concurrent_entry(abb_env):
    solver = KDLStateSolver(abb_env.getSceneGraph())
    with pytest.raises(AttributeError, match="getStateConcurrent"):
        solver.getStateConcurrent({})
