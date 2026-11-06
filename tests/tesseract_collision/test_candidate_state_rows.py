"""Per-candidate native collision decisions retain row order and input ownership."""

import numpy as np
import pytest
from test_batch_state_checks import reference

from tesseract_robotics import tesseract_collision as collision

pytest_plugins = ("test_batch_state_checks",)


def test_all_ordered_rows_match_independent_scalar_checks(cell):
    _env, solver, manager, names, limits, q = cell
    invalid = q.copy()
    invalid[0] = limits[0, 1] + 1
    rows = np.asarray([q, invalid, q])
    expected = [reference(solver, manager, names, limits, row[None, :])[0] for row in rows]
    assert collision.classify_candidate_state_rows(solver, manager, names, rows, limits) == expected


def test_malformed_and_unknown_validator_fail_loudly(cell):
    _env, solver, manager, names, limits, q = cell
    with pytest.raises(collision.InvalidBatch):
        collision.classify_candidate_state_rows(solver, manager, names, q[None, :-1], limits)
    original = manager.getContactAllowedValidator()
    from tesseract_robotics.tesseract_common import CombinedContactAllowedValidator

    manager.setContactAllowedValidator(CombinedContactAllowedValidator())
    try:
        with pytest.raises(collision.UnsupportedBatchContext):
            collision.classify_candidate_state_rows(solver, manager, names, q[None, :], limits)
    finally:
        manager.setContactAllowedValidator(original)


def test_input_copy_and_gil_progress_on_owned_scene(cell):
    from threading import Event, Thread
    from time import sleep

    env, _solver, _manager, names, limits, q = cell
    solver = env.getStateSolver()
    manager = env.getDiscreteContactManager()
    rows = np.repeat(q[None, :], 20000, axis=0)
    mutated = Event()
    progressed = Event()

    def mutate():
        sleep(0.01)
        rows[:] = np.nan
        progressed.set()
        mutated.set()

    thread = Thread(target=mutate)
    thread.start()
    try:
        decisions = collision.classify_candidate_state_rows(solver, manager, names, rows, limits)
        mutated_in_call = mutated.is_set()
    finally:
        thread.join()
    assert progressed.is_set() and mutated_in_call
    assert len(decisions) == len(rows)
    assert set(decisions) == {"clear"}


def test_acm_and_nonnegative_contacts_keep_all_rows(cell):
    from tesseract_robotics.tesseract_common import (
        ACMContactAllowedValidator,
        AllowedCollisionMatrix,
        CollisionMarginData,
    )

    _env, solver, manager, names, limits, q = cell
    original = manager.getContactAllowedValidator()
    original_margin = manager.getCollisionMarginData()
    try:
        manager.setCollisionMarginData(CollisionMarginData(0.05))
        results = collision.ContactResultMap()
        manager.setCollisionObjectsTransform(
            solver.getState(dict(zip(names, q, strict=True))).link_transforms
        )
        manager.contactTest(results, collision.ContactRequest(collision.ContactTestType.ALL))
        contacts = collision.ContactResultVector()
        results.flattenMoveResults(contacts)
        assert any(hit.distance < 0 for hit in contacts)
        acm = AllowedCollisionMatrix()
        for hit in contacts:
            if hit.distance < 0:
                acm.addAllowedCollision(*map(str, hit.link_ids), "candidate test")
        manager.setContactAllowedValidator(ACMContactAllowedValidator(acm))
        rows = np.stack([q, q])
        expected = [reference(solver, manager, names, limits, row[None, :])[0] for row in rows]
        assert (
            collision.classify_candidate_state_rows(solver, manager, names, rows, limits)
            == expected
        )
    finally:
        manager.setContactAllowedValidator(original)
        manager.setCollisionMarginData(original_margin)
