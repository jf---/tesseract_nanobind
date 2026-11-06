"""Ordered native pose residuals preserve scalar fixture-relative semantics."""

from threading import Event, Thread
from time import sleep

import numpy as np
import pytest

from tesseract_robotics import tesseract_state_solver as state_solver
from tesseract_robotics.tesseract_common import Quaterniond

pytest_plugins = ("test_batch_state_checks",)


def scalar(solver, names, held_names, rows, targets, workpiece, tcp):
    residuals = []
    for q, target in zip(rows, targets, strict=True):
        values = dict(zip(names, q, strict=True))
        held = {name: values[name] for name in held_names}
        workpiece_world = np.asarray(solver.getState(held).link_transforms[workpiece].matrix)
        tcp_world = np.asarray(solver.getState(values).link_transforms[tcp].matrix)
        target_world = workpiece_world @ target.reshape(4, 4)
        distance = np.linalg.norm(tcp_world[:3, 3] - target_world[:3, 3])
        angle = Quaterniond(tcp_world[:3, :3]).angularDistance(Quaterniond(target_world[:3, :3]))
        residuals.append((distance, angle))
    return np.asarray(residuals)


def test_fixture_aba_and_zero_and_near_pi_rotations(cell):
    _env, solver, _manager, names, limits, q = cell
    held_names = ("positioner_joint_1",)
    pos = names.index(held_names[0])
    b = q.copy()
    b[pos] = min(limits[pos, 1], q[pos] + 0.2)
    assert b[pos] != q[pos]
    rows = np.asarray([q, b, q])
    identity = np.eye(4)
    near_pi = np.eye(4)
    angle = np.pi - 1e-6
    near_pi[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    targets = np.asarray([identity, near_pi, identity]).reshape(-1, 16)
    expected = scalar(solver, names, held_names, rows, targets, "base_link", "link_6")
    actual = state_solver.workpiece_pose_residual_rows(
        solver, names, held_names, rows, targets, "base_link", "link_6"
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual[0], actual[2])


def test_invalid_shapes_names_links_rotations_and_solver_fail_loudly(cell):
    env, solver, _manager, names, _limits, q = cell
    rows = q[None, :]
    target = np.eye(4).reshape(1, 16)
    calls = [
        (solver, names, (), rows[:, :-1], target, "base_link", "link_6"),
        (solver, names, (), rows, target[:, :-1], "base_link", "link_6"),
        (solver, (*names[:-1], names[0]), (), rows, target, "base_link", "link_6"),
        (solver, names, ("unknown_joint",), rows, target, "base_link", "link_6"),
        (solver, names, (), rows, target, "missing_link", "link_6"),
    ]
    bad_rotation = target.copy()
    bad_rotation[0, 0] = 2.0
    calls.append((solver, names, (), rows, bad_rotation, "base_link", "link_6"))
    for args in calls:
        with pytest.raises(state_solver.PoseBatchInputError):
            state_solver.workpiece_pose_residual_rows(*args)
    kdl = state_solver.KDLStateSolver(env.getSceneGraph())
    with pytest.raises(state_solver.PoseBatchContextError):
        state_solver.workpiece_pose_residual_rows(
            kdl, names, (), rows, target, "base_link", "link_6"
        )


def test_copies_rows_and_targets_before_releasing_gil(cell):
    env, _, _, names, _, q = cell
    solver = env.getStateSolver()
    rows = np.repeat(q[None, :], 20000, axis=0)
    targets = np.repeat(np.eye(4).reshape(1, 16), len(rows), axis=0)
    original = scalar(solver, names, (), rows[:1], targets[:1], "base_link", "link_6")[0]
    changed = Event()

    def mutate():
        sleep(0.01)
        rows[:] = np.nan
        targets[:] = np.nan
        changed.set()

    thread = Thread(target=mutate)
    thread.start()
    try:
        actual = state_solver.workpiece_pose_residual_rows(
            solver, names, (), rows, targets, "base_link", "link_6"
        )
        assert changed.is_set()
    finally:
        thread.join()
    np.testing.assert_allclose(actual, np.tile(original, (len(rows), 1)), rtol=0, atol=1e-12)


def test_independently_owned_solvers_can_run_in_parallel(cell):
    from concurrent.futures import ThreadPoolExecutor

    env, _, _, names, _, q = cell
    rows = np.repeat(q[None, :], 1000, axis=0)
    targets = np.repeat(np.eye(4).reshape(1, 16), len(rows), axis=0)
    first, second = env.getStateSolver(), env.getStateSolver()
    assert first is not second

    def run(solver):
        return state_solver.workpiece_pose_residual_rows(
            solver, names, (), rows, targets, "base_link", "link_6"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.map(run, (first, second))
    np.testing.assert_array_equal(a, b)


def test_workpiece_fk_uses_only_held_values_not_planned_values(cell):
    _env, solver, _manager, names, _, q = cell
    moved = q.copy()
    moved[0] += 0.4
    rows = np.asarray([q, moved])
    targets = np.repeat(np.eye(4).reshape(1, 16), len(rows), axis=0)
    expected = scalar(solver, names, (), rows, targets, "link_6", "base_link")
    actual = state_solver.workpiece_pose_residual_rows(
        solver, names, (), rows, targets, "link_6", "base_link"
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_nonfinite_residual_is_rejected(cell):
    _env, solver, _manager, names, _, q = cell
    target = np.eye(4)
    target[0, 3] = np.finfo(float).max / 2
    with pytest.raises(state_solver.PoseBatchInputError, match="nonfinite residual"):
        state_solver.workpiece_pose_residual_rows(
            solver, names, (), q[None, :], target.reshape(1, 16), "base_link", "link_6"
        )
