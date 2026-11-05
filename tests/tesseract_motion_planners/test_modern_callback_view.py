"""The native-owned callback view remains guarded for modern QP methods."""

import pytest

from tesseract_robotics import trajopt_sqp as tsqp
from tesseract_robotics.tesseract_motion_planners_trajopt_ifopt import (
    TrajOptIfoptMotionPlanner,
    TrajOptIfoptOSQPSolverProfile,
)
from tests.tesseract_motion_planners.test_trajopt_ifopt_planner import (
    TRAJOPT_IFOPT_NAMESPACE,
    _request,
)

pytest_plugins = ["tests.tesseract_motion_planners.test_trajopt_ifopt_planner"]


def test_native_callback_view_forwards_and_expires(kuka_iiwa_environment):
    env, manip_info, joint_names = kuka_iiwa_environment

    class Callback(tsqp.SQPCallback):
        def __init__(self):
            super().__init__()
            self.views = []

        def execute(self, problem, _results):
            problem.setSumConstraintSetViolations(False)
            self.views.append(problem)
            return True

    callback = Callback()
    profile = TrajOptIfoptOSQPSolverProfile()
    profile.opt_params.max_iter = 5
    profile.callbacks = [callback]
    TrajOptIfoptMotionPlanner(TRAJOPT_IFOPT_NAMESPACE).solve(
        _request(env, manip_info, joint_names, profile)
    )
    assert callback.views
    for view in callback.views:
        with pytest.raises(tsqp.QPProblemExpiredError):
            view.setSumConstraintSetViolations(True)
