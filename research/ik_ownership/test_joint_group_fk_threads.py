"""Threading and input contracts for a real native JointGroup FK call."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tesseract_robotics.planning import Robot
from tests.test_gil_entry_points import assert_releases_gil

pytest_plugins = ["tests.tesseract_time_parameterization.test_time_parameterization"]


def test_fk_releases_gil_during_native_call(abb_env):
    group = abb_env.getJointGroup("manipulator")
    joints = np.zeros(group.numJoints())
    assert_releases_gil(lambda: group.calcFwdKin(joints))


def test_independent_groups_fk_match_serial(abb_env):
    other_robot = Robot.from_tesseract_support("abb_irb2400")
    groups = [abb_env.getJointGroup("manipulator"), other_robot.env.getJointGroup("manipulator")]
    joints = [np.zeros(groups[0].numJoints()), np.full(groups[1].numJoints(), 0.1)]
    expected = [group.calcFwdKin(q)["tool0"].matrix.copy() for group, q in zip(groups, joints)]

    def run(index):
        group = groups[index]
        for _ in range(100):
            actual = group.calcFwdKin(joints[index])["tool0"].matrix
            np.testing.assert_allclose(actual, expected[index], atol=0, rtol=0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(2)))


def test_fk_rejects_wrong_input_dimension(abb_env):
    group = abb_env.getJointGroup("manipulator")
    with pytest.raises(ValueError, match="joint vector size"):
        group.calcFwdKin(np.zeros(group.numJoints() + 1))
