"""Captured Ruckig update trajectories are the smoother's own native outputs."""

import numpy as np
import pytest

from tesseract_robotics.planning import Robot
from tesseract_robotics.tesseract_command_language import (
    CompositeInstruction,
    MoveInstruction,
    MoveInstructionPoly_wrap_MoveInstruction,
    MoveInstructionType,
    ProfileDictionary,
    StateWaypoint,
    StateWaypointPoly_wrap_StateWaypoint,
)
from tesseract_robotics.tesseract_common import ManipulatorInfo
from tesseract_robotics.tesseract_time_parameterization import (
    InstructionsTrajectory,
    RuckigSolveError,
    RuckigTimeError,
    RuckigTrajectorySmoothing,
    RuckigTrajectorySmoothingCompositeProfile,
)

ABB_JOINTS = [f"joint_{i}" for i in range(1, 7)]


@pytest.fixture(scope="module")
def abb_env():
    return Robot.from_tesseract_support("abb_irb2400").env


def _manipulator_info():
    info = ManipulatorInfo()
    info.manipulator = "manipulator"
    info.working_frame = "base_link"
    info.tcp_frame = "tool0"
    return info


def timed_program(times):
    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(_manipulator_info())
    for i, time in enumerate(times):
        q = np.zeros(6)
        q[0] = i * 0.2
        velocity = np.zeros(6)
        velocity[0] = 0.1
        waypoint = StateWaypoint(ABB_JOINTS, q, velocity, np.zeros(6), time)
        program.appendMoveInstruction(
            MoveInstructionPoly_wrap_MoveInstruction(
                MoveInstruction(
                    StateWaypointPoly_wrap_StateWaypoint(waypoint),
                    MoveInstructionType.FREESPACE,
                    "DEFAULT",
                )
            )
        )
    return program


def profiles(max_factor=100.0):
    profile = RuckigTrajectorySmoothingCompositeProfile(1.5, max_factor)
    profile.override_limits = True
    profile.velocity_limits = np.column_stack((-np.full(6, 2.0), np.full(6, 2.0)))
    profile.acceleration_limits = np.column_stack((-np.full(6, 4.0), np.full(6, 4.0)))
    profile.jerk_limits = np.column_stack((-np.full(6, 8.0), np.full(6, 8.0)))
    result = ProfileDictionary()
    result.addProfile("RuckigTrajectorySmoothing", "DEFAULT", profile)
    return result


def test_captured_segments_are_successful_final_attempt_with_update_states(abb_env):
    program = timed_program([0.0, 0.01, 0.04])
    reference = timed_program([0.0, 0.01, 0.04])
    smoother = RuckigTrajectorySmoothing()
    assert smoother.compute(reference, abb_env, profiles())
    segments = smoother.compute_with_trajectories(program, abb_env, profiles())
    actual, baseline = InstructionsTrajectory(program), InstructionsTrajectory(reference)
    assert actual.getTimeFromStart(2) > 0.04
    assert len(segments) == actual.size() - 1
    for i, segment in enumerate(segments):
        assert (segment.start_index, segment.end_index) == (i, i + 1)
        assert segment.waypoint_start_time == pytest.approx(actual.getTimeFromStart(i))
        assert segment.waypoint_end_time == pytest.approx(actual.getTimeFromStart(i + 1))
        assert segment.update_time > segment.duration
        np.testing.assert_allclose(segment.evaluate_update(), segment.update_state, atol=1e-10)
        assert len(segment.evaluate(segment.duration)) == 4
        for getter in ("getPosition", "getVelocity", "getAcceleration"):
            np.testing.assert_allclose(
                getattr(actual, getter)(i), getattr(baseline, getter)(i), atol=1e-10
            )
        assert actual.getTimeFromStart(i) == pytest.approx(baseline.getTimeFromStart(i))
        with pytest.raises(RuckigTimeError):
            segment.evaluate(segment.duration + 1.0)


def test_invalid_clock_and_failed_attempt_raise(abb_env):
    smoother = RuckigTrajectorySmoothing()
    with pytest.raises(RuckigSolveError):
        smoother.compute_with_trajectories(timed_program([0.0, 0.02, 0.01]), abb_env, profiles())
    with pytest.raises(RuckigSolveError):
        smoother.compute_with_trajectories(
            timed_program([0.0, 0.001, 0.002]), abb_env, profiles(1.01)
        )


def test_checked_smoother_rejects_nonuniform_false_success_without_mutating_input(abb_env):
    from tesseract_robotics.tesseract_time_parameterization import RuckigSegmentTimingError

    unsafe = timed_program([0.0, 0.001, 10.0])
    baseline = timed_program([0.0, 0.001, 10.0])
    smoother = RuckigTrajectorySmoothing()
    assert smoother.compute(baseline, abb_env, profiles())
    assert InstructionsTrajectory(baseline).getTimeFromStart(1) == pytest.approx(0.001)
    with pytest.raises(RuckigSegmentTimingError, match=r"segment 0:.*waypoint interval"):
        smoother.compute_checked_trajectories(unsafe, abb_env, profiles())
    original = InstructionsTrajectory(unsafe)
    assert [original.getTimeFromStart(i) for i in range(3)] == pytest.approx([0.0, 0.001, 10.0])
    np.testing.assert_allclose(original.getPosition(1), [0.2, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(original.getVelocity(1), [0.1, 0, 0, 0, 0, 0])


def test_checked_smoother_accepts_a_matching_native_interval(abb_env):
    from tesseract_robotics.tesseract_time_parameterization import RuckigTrajectory

    q0 = [0.0] * 6
    q1 = [0.2, 0, 0, 0, 0, 0]
    v = [0.1, 0, 0, 0, 0, 0]
    zero = [0.0] * 6
    direct = RuckigTrajectory.solve(q0, q1, v, v, zero, zero, [2.0] * 6, [4.0] * 6, [8.0] * 6, 0.0)
    duration = direct.duration
    for _ in range(4):
        duration = float(np.nextafter(duration, np.inf))
    program = timed_program([0.0, duration])
    segments = RuckigTrajectorySmoothing().compute_checked_trajectories(
        program, abb_env, profiles()
    )
    assert len(segments) == 1
    assert segments[0].duration == pytest.approx(duration, rel=1e-12)
    assert InstructionsTrajectory(program).getTimeFromStart(1) == duration


def test_checked_smoother_rejects_implicit_retiming_and_preserves_program(abb_env):
    from tesseract_robotics.tesseract_time_parameterization import RuckigSegmentTimingError

    program = timed_program([0.0, 0.02, 0.04])
    before = InstructionsTrajectory(program)
    time_before = [before.getTimeFromStart(i) for i in range(before.size())]
    velocity_before = [before.getVelocity(i).copy() for i in range(before.size())]
    with pytest.raises(RuckigSegmentTimingError, match="smoother changed the input clock"):
        RuckigTrajectorySmoothing().compute_checked_trajectories(program, abb_env, profiles())
    after = InstructionsTrajectory(program)
    assert [after.getTimeFromStart(i) for i in range(after.size())] == time_before
    for i in range(after.size()):
        np.testing.assert_array_equal(after.getVelocity(i), velocity_before[i])


def retime_limits():
    return {
        "position_lower": [-5.0] * 6,
        "position_upper": [5.0] * 6,
        "max_velocity": [2.0] * 6,
        "max_acceleration": [4.0] * 6,
        "max_jerk": [8.0] * 6,
    }


def test_explicit_freespace_retiming_preserves_states_and_aligns_actual_curves():
    from tesseract_robotics.tesseract_time_parameterization import retime_freespace

    program = timed_program([0.0, 0.001, 10.0])
    original = InstructionsTrajectory(program)
    before = [original.getTimeFromStart(i) for i in range(original.size())]
    retimed, segments = retime_freespace(program, **retime_limits())
    result = InstructionsTrajectory(retimed)
    assert [original.getTimeFromStart(i) for i in range(original.size())] == before
    assert result.getTimeFromStart(1) > before[1]
    assert len(segments) == result.size() - 1
    for i, segment in enumerate(segments):
        assert (segment.start_index, segment.end_index) == (i, i + 1)
        assert segment.start_time == pytest.approx(result.getTimeFromStart(i))
        assert segment.end_time == pytest.approx(result.getTimeFromStart(i + 1))
        assert segment.duration == pytest.approx(segment.end_time - segment.start_time)
        assert segment.original_duration == pytest.approx(before[i + 1] - before[i])
        for endpoint, waypoint in (
            (segment.evaluate(0.0), i),
            (segment.evaluate(segment.duration), i + 1),
        ):
            np.testing.assert_allclose(endpoint[0], original.getPosition(waypoint), atol=1e-10)
            np.testing.assert_allclose(endpoint[1], original.getVelocity(waypoint), atol=1e-10)
            np.testing.assert_allclose(endpoint[2], original.getAcceleration(waypoint), atol=1e-10)
            np.testing.assert_array_equal(
                result.getPosition(waypoint), original.getPosition(waypoint)
            )
            np.testing.assert_array_equal(
                result.getVelocity(waypoint), original.getVelocity(waypoint)
            )
            np.testing.assert_array_equal(
                result.getAcceleration(waypoint), original.getAcceleration(waypoint)
            )


def test_freespace_retiming_refuses_process_events_and_bad_limits():
    from tesseract_robotics.tesseract_command_language import WaitInstruction
    from tesseract_robotics.tesseract_time_parameterization import (
        RuckigFreespaceError,
        retime_freespace,
    )

    program = timed_program([0.0, 1.0, 2.0])
    program.push_back(WaitInstruction(1.0))
    with pytest.raises(RuckigFreespaceError, match="event"):
        retime_freespace(program, **retime_limits())
    invalid = retime_limits()
    invalid["max_jerk"] = [0.0] * 6
    with pytest.raises(RuckigFreespaceError, match="limit"):
        retime_freespace(timed_program([0.0, 1.0]), **invalid)


def test_freespace_retiming_rejects_actual_native_position_overshoot():
    from tesseract_robotics.tesseract_time_parameterization import (
        RuckigFreespaceError,
        retime_freespace,
    )

    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(_manipulator_info())
    for i, (position0, velocity0) in enumerate(((0.1, -1.0), (0.2, 0.0))):
        q = np.zeros(6)
        v = np.zeros(6)
        q[0], v[0] = position0, velocity0
        waypoint = StateWaypoint(ABB_JOINTS, q, v, np.zeros(6), float(i))
        program.appendMoveInstruction(
            MoveInstructionPoly_wrap_MoveInstruction(
                MoveInstruction(
                    StateWaypointPoly_wrap_StateWaypoint(waypoint),
                    MoveInstructionType.FREESPACE,
                    "DEFAULT",
                )
            )
        )
    limits = retime_limits()
    limits["position_lower"][0] = 0.0
    limits["position_upper"][0] = 0.3
    limits["max_acceleration"][0] = 1.0
    limits["max_jerk"][0] = 2.0
    with pytest.raises(RuckigFreespaceError, match="position extrema"):
        retime_freespace(program, **limits)


def test_freespace_retiming_refuses_linear_motion():
    from tesseract_robotics.tesseract_time_parameterization import (
        RuckigFreespaceError,
        retime_freespace,
    )

    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(_manipulator_info())
    for i in range(2):
        waypoint = StateWaypoint(
            ABB_JOINTS, np.full(6, 0.1 * i), np.zeros(6), np.zeros(6), float(i)
        )
        program.appendMoveInstruction(
            MoveInstructionPoly_wrap_MoveInstruction(
                MoveInstruction(
                    StateWaypointPoly_wrap_StateWaypoint(waypoint),
                    MoveInstructionType.LINEAR,
                    "DEFAULT",
                )
            )
        )
    with pytest.raises(RuckigFreespaceError, match="FREESPACE"):
        retime_freespace(program, **retime_limits())


def test_freespace_retiming_copies_program_before_releasing_gil():
    import sys
    import threading

    from tesseract_robotics.tesseract_command_language import WaitInstruction
    from tesseract_robotics.tesseract_time_parameterization import retime_freespace

    program = timed_program([i * 0.02 for i in range(2001)])
    limits = retime_limits()
    limits["position_lower"] = [-1000.0] * 6
    limits["position_upper"] = [1000.0] * 6
    gate = threading.Lock()
    ready = threading.Event()
    progressed = threading.Event()
    gate.acquire()

    def change_input():
        ready.set()
        with gate:
            program.push_back(WaitInstruction(1.0))
            progressed.set()

    worker = threading.Thread(target=change_input)
    worker.start()
    assert ready.wait(5.0)
    old_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(100.0)
        gate.release()
        retimed, segments = retime_freespace(program, **limits)
        ran_during_solve = progressed.is_set()
    finally:
        sys.setswitchinterval(old_interval)
        worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert ran_during_solve
    assert len(segments) == 2000
    assert InstructionsTrajectory(retimed).size() == 2001


def test_freespace_retiming_rejects_duplicate_joint_ids_and_infeasible_boundaries():
    from tesseract_robotics.tesseract_time_parameterization import (
        RuckigFreespaceError,
        retime_freespace,
    )

    names = ["joint_1"] * 6
    program = CompositeInstruction("DEFAULT")
    for i in range(2):
        waypoint = StateWaypoint(names, np.full(6, 0.1 * i), np.zeros(6), np.zeros(6), float(i))
        program.appendMoveInstruction(
            MoveInstructionPoly_wrap_MoveInstruction(
                MoveInstruction(
                    StateWaypointPoly_wrap_StateWaypoint(waypoint),
                    MoveInstructionType.FREESPACE,
                    "DEFAULT",
                )
            )
        )
    with pytest.raises(RuckigFreespaceError, match="distinct"):
        retime_freespace(program, **retime_limits())
    limits = retime_limits()
    limits["max_velocity"][0] = 0.05
    with pytest.raises(RuckigFreespaceError, match="segment"):
        retime_freespace(timed_program([0.0, 1.0]), **limits)
