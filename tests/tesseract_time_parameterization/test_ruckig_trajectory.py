"""Consumer contracts for independently generated Ruckig trajectories."""

import numpy as np
import pytest

from tesseract_robotics.tesseract_time_parameterization import (
    RuckigInputError,
    RuckigTimeError,
    RuckigTrajectory,
)


def inputs():
    return {
        "start_position": [0.0, 0.2],
        "target_position": [1.0, -0.3],
        "start_velocity": [0.2, 0.0],
        "target_velocity": [0.1, 0.0],
        "start_acceleration": [0.0, 0.0],
        "target_acceleration": [0.0, 0.0],
        "max_velocity": [2.0, 2.0],
        "max_acceleration": [4.0, 4.0],
        "max_jerk": [8.0, 8.0],
        "minimum_duration": 2.0,
    }


def test_owned_trajectory_evaluates_actual_state_and_dynamic_bounds():
    spec = inputs()
    trajectory = RuckigTrajectory.solve(**spec)
    assert trajectory.duration >= spec["minimum_duration"]
    start = trajectory.evaluate(0.0)
    end = trajectory.evaluate(trajectory.duration)
    for actual, key in zip(start[:3], ("start_position", "start_velocity", "start_acceleration")):
        np.testing.assert_allclose(actual, spec[key], atol=1e-10)
    for actual, key in zip(end[:3], ("target_position", "target_velocity", "target_acceleration")):
        np.testing.assert_allclose(actual, spec[key], atol=1e-10)
    for time in np.linspace(0.0, trajectory.duration, 201):
        position, velocity, acceleration, jerk = trajectory.evaluate(float(time))
        assert np.isfinite(position).all()
        assert np.all(np.abs(velocity) <= np.asarray(spec["max_velocity"]) + 1e-10)
        assert np.all(np.abs(acceleration) <= np.asarray(spec["max_acceleration"]) + 1e-10)
        assert np.all(np.abs(jerk) <= np.asarray(spec["max_jerk"]) + 1e-10)
    spec["start_position"][0] = 55.0
    np.testing.assert_allclose(trajectory.evaluate(0.0)[0], [0.0, 0.2])


def test_invalid_limits_and_shape_fail_loudly():
    spec = inputs()
    spec["max_jerk"] = [0.0, 1.0]
    with pytest.raises(RuckigInputError):
        RuckigTrajectory.solve(**spec)
    spec = inputs()
    spec["target_position"] = [1.0]
    with pytest.raises(RuckigInputError):
        RuckigTrajectory.solve(**spec)
    spec = inputs()
    spec["minimum_duration"] = float("nan")
    with pytest.raises(RuckigInputError):
        RuckigTrajectory.solve(**spec)


def test_evaluation_rejects_times_outside_actual_trajectory():
    trajectory = RuckigTrajectory.solve(**inputs())
    for time in (-1.0, trajectory.duration + 1.0, float("nan")):
        with pytest.raises(RuckigTimeError):
            trajectory.evaluate(time)


def test_solve_copies_inputs_and_releases_gil_while_native_work_runs():
    import sys
    import threading

    dof = 100000
    spec = {
        "start_position": [0.0] * dof,
        "target_position": [0.5] * dof,
        "start_velocity": [0.0] * dof,
        "target_velocity": [0.0] * dof,
        "start_acceleration": [0.0] * dof,
        "target_acceleration": [0.0] * dof,
        "max_velocity": [2.0] * dof,
        "max_acceleration": [4.0] * dof,
        "max_jerk": [8.0] * dof,
        "minimum_duration": 2.0,
    }
    gate = threading.Lock()
    ready = threading.Event()
    progressed = threading.Event()
    gate.acquire()

    def mutate_after_release():
        ready.set()
        with gate:
            spec["target_position"][0] = 99.0
            progressed.set()

    worker = threading.Thread(target=mutate_after_release)
    worker.start()
    assert ready.wait(5.0)
    old_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(100.0)
        gate.release()
        trajectory = RuckigTrajectory.solve(**spec)
        ran_during_solve = progressed.is_set()
    finally:
        sys.setswitchinterval(old_interval)
        worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert ran_during_solve
    assert trajectory.evaluate(trajectory.duration)[0][0] == pytest.approx(0.5)
