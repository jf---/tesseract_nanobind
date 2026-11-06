"""Collision-only row decisions do not short-circuit on joint limits."""

import numpy as np
import pytest

from tesseract_robotics import tesseract_collision as collision

pytest_plugins = ("test_batch_state_checks",)


def scalar_collision(solver, manager, names, q):
    state = solver.getState(dict(zip(names, q, strict=True)))
    manager.setCollisionObjectsTransform(state.link_transforms)
    contacts = collision.ContactResultMap()
    manager.contactTest(contacts, collision.ContactRequest(collision.ContactTestType.ALL))
    flat = collision.ContactResultVector()
    contacts.flattenMoveResults(flat)
    return "sampled_collision" if any(hit.distance < 0 for hit in flat) else "clear"


def test_collisions_are_checked_even_outside_joint_limits(cell):
    _env, solver, manager, names, limits, q = cell
    outside = q.copy()
    outside[0] = limits[0, 1] + 0.1
    rows = np.asarray([q, outside, q])
    expected = [scalar_collision(solver, manager, names, row) for row in rows]
    assert expected[0] == "sampled_collision"
    assert collision.classify_collision_state_rows_first(solver, manager, names, rows) == expected
    assert (
        collision.classify_candidate_state_rows_first(solver, manager, names, rows, limits)[1]
        == "joint_limit"
    )


def test_malformed_rows_and_unsupported_validator_fail(cell):
    from tesseract_robotics.tesseract_common import CombinedContactAllowedValidator

    _env, solver, manager, names, _limits, q = cell
    with pytest.raises(collision.InvalidBatch):
        collision.classify_collision_state_rows_first(solver, manager, names, q[None, :-1])
    with pytest.raises(collision.InvalidBatch):
        collision.classify_collision_state_rows_first(
            solver, manager, names, np.full((1, len(q)), np.nan)
        )
    original = manager.getContactAllowedValidator()
    manager.setContactAllowedValidator(CombinedContactAllowedValidator())
    try:
        with pytest.raises(collision.UnsupportedBatchContext):
            collision.classify_collision_state_rows_first(solver, manager, names, q[None, :])
    finally:
        manager.setContactAllowedValidator(original)


def test_aba_fixture_and_acm_preserve_order(cell):
    from tesseract_robotics.tesseract_common import (
        ACMContactAllowedValidator,
        AllowedCollisionMatrix,
    )

    _env, solver, manager, names, limits, q = cell
    b = q.copy()
    positioner = names.index("positioner_joint_1")
    b[positioner] = min(limits[positioner, 1], q[positioner] + 0.2)
    assert b[positioner] != q[positioner]
    rows = np.asarray([q, b, q])
    original = manager.getContactAllowedValidator()
    try:
        assert collision.classify_collision_state_rows_first(solver, manager, names, rows) == [
            scalar_collision(solver, manager, names, row) for row in rows
        ]
        acm = AllowedCollisionMatrix()
        acm.addAllowedCollision("batch_test_obstacle", "link_6", "test")
        manager.setContactAllowedValidator(ACMContactAllowedValidator(acm))
        assert collision.classify_collision_state_rows_first(solver, manager, names, rows) == [
            scalar_collision(solver, manager, names, row) for row in rows
        ]
    finally:
        manager.setContactAllowedValidator(original)


def test_copied_rows_and_python_progress_during_native_call(cell):
    from threading import Event, Thread
    from time import sleep

    env, _, _, names, _limits, q = cell
    solver = env.getStateSolver()
    manager = env.getDiscreteContactManager()
    rows = np.repeat(q[None, :], 20000, axis=0)
    mutated = Event()

    def mutate():
        sleep(0.01)
        rows[:] = np.nan
        mutated.set()

    thread = Thread(target=mutate)
    thread.start()
    try:
        decisions = collision.classify_collision_state_rows_first(solver, manager, names, rows)
        assert mutated.is_set()
    finally:
        thread.join()
    assert len(decisions) == len(rows)
    assert set(decisions) == {"clear"}
