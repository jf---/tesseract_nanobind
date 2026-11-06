"""Native inverse solver reads a snapshot even when Python mutates its inputs."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from ik_ownership_probe import BlockingInverse

from tesseract_robotics import tesseract_kinematics  # noqa: F401
from tesseract_robotics.tesseract_common import Isometry3d
from tesseract_robotics.tesseract_kinematics import (
    KinematicGroup,
    KinGroupIKInput,
    KinGroupIKInputs,
)

pytest_plugins = ["tests.tesseract_time_parameterization.test_time_parameterization"]


def test_inverse_seed_is_owned_during_native_call():
    solver = BlockingInverse()
    seed = np.array([0.25])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(solver.calcInvKin, {"tool0": Isometry3d.Identity()}, seed)
        try:
            assert solver.await_entry()
            seed[0] = 0.75
        finally:
            solver.unblock()
        result = future.result()
    np.testing.assert_array_equal(np.asarray(result[0]).reshape(-1), [0.25])


def test_inverse_exception_propagates_after_gil_release():
    solver = BlockingInverse()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(solver.calcInvKin, {"tool0": Isometry3d.Identity()}, np.array([0.25]))
        try:
            assert solver.await_entry()
        finally:
            solver.fail()
        with pytest.raises(RuntimeError, match="probe solver failure"):
            future.result()


@pytest.mark.parametrize("form", ["single", "opaque", "list", "list_alias"])
def test_group_ik_snapshots_inputs_during_native_call(abb_env, form):
    names = list(abb_env.getJointGroup("manipulator").getJointNames())
    solver = BlockingInverse(names)
    group = KinematicGroup("test", names, solver, abb_env.getSceneGraph(), abb_env.getState())
    seed = np.full(len(names), 0.25)
    pose = KinGroupIKInput(Isometry3d.Identity(), "base_link", "tool0")
    opaque = KinGroupIKInputs()
    opaque.append(pose)
    poses = [pose]
    argument = {"single": pose, "opaque": opaque, "list": poses, "list_alias": poses}[form]
    method = group.calcInvKinMultiple if form == "list_alias" else group.calcInvKin
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(method, argument, seed)
        try:
            assert solver.await_entry()
            seed[:] = 0.75
            opaque.clear()
            poses.clear()
        finally:
            solver.unblock()
        result = future.result()
    # Joint harmonization may round by a few floating-point ulps.
    rounding_tolerance = 4 * np.finfo(float).eps
    np.testing.assert_allclose(
        np.asarray(result[0]).reshape(-1),
        np.full(len(names), 0.25),
        rtol=0,
        atol=rounding_tolerance,
    )
