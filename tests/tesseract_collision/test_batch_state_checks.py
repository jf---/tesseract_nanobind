"""Ordered native batch checks against the standard robot cell fixture."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread
from time import sleep

import numpy as np
import pytest

from tesseract_robotics.tesseract_collision import (
    ContactRequest,
    ContactResultMap,
    ContactResultVector,
    ContactTestType,
    InvalidBatch,
    UnsupportedBatchContext,
    check_batched_state_rows,
)
from tesseract_robotics.tesseract_common import (
    ACMContactAllowedValidator,
    AllowedCollisionMatrix,
    CollisionMarginData,
    CombinedContactAllowedValidator,
    GeneralResourceLocator,
    Isometry3d,
    VectorIsometry3d,
)
from tesseract_robotics.tesseract_environment import Environment
from tesseract_robotics.tesseract_geometry import GeometriesConst, Sphere

# Long enough for another Python thread to run during the native fixture loop.
THREAD_PROGRESS_ROWS = 20000


@pytest.fixture(scope="module")
def cell():
    locator = GeneralResourceLocator()
    urdf = locator.locateResource("package://tesseract/support/urdf/abb_irb2400_on_positioner.urdf")
    srdf = locator.locateResource("package://tesseract/support/urdf/abb_irb2400_on_positioner.srdf")
    assert urdf is not None and srdf is not None
    env = Environment()
    assert env.init(Path(urdf.getFilePath()), Path(srdf.getFilePath()), locator)
    solver = env.getStateSolver()
    manager = env.getDiscreteContactManager()
    names = tuple(str(name) for name in solver.getActiveJointIds())
    limits = np.asarray(solver.getLimits().joint_limits)
    state = env.getState()
    q = np.array([float(state.joints[name]) for name in names])
    # The manager owns this obstacle. Its known overlap with link_6 guarantees a contact.
    shape = GeometriesConst()
    shape.append(Sphere(0.15))
    pose = np.eye(4)
    pose[:3, 3] = np.asarray(state.link_transforms["link_6"].translation)
    poses = VectorIsometry3d()
    poses.append(Isometry3d(pose))
    assert manager.addCollisionObject("batch_test_obstacle", 0, shape, poses)
    manager.setActiveCollisionObjects([*manager.getActiveCollisionObjects(), "batch_test_obstacle"])
    manager.setCollisionMarginData(CollisionMarginData(0.0))
    return env, solver, manager, names, limits, q


def reference(solver, manager, names, limits, rows):
    for index, row in enumerate(rows):
        if np.any(row < limits[:, 0]) or np.any(row > limits[:, 1]):
            return ("joint_limit", index, "", "", 0.0)
        state = solver.getState(dict(zip(names, row, strict=True)))
        manager.setCollisionObjectsTransform(state.link_transforms)
        contacts = ContactResultMap()
        manager.contactTest(contacts, ContactRequest(ContactTestType.ALL))
        flat = ContactResultVector()
        contacts.flattenMoveResults(flat)
        negative = [hit for hit in flat if hit.distance < 0]
        if negative:
            hit = min(negative, key=lambda item: item.distance)
            return (
                "sampled_collision",
                index,
                *sorted(map(str, hit.link_ids)),
                float(hit.distance),
            )
    return ("clear", -1, "", "", 0.0)


def test_ordered_collision_and_limits(cell):
    _env, solver, manager, names, limits, q = cell
    invalid = q.copy()
    invalid[0] = limits[0, 1] + 1.0
    assert reference(solver, manager, names, limits, q[None, :])[0] == "sampled_collision"
    for rows in (np.array([q]), np.array([q, invalid]), np.array([invalid, q])):
        assert check_batched_state_rows(solver, manager, names, rows, limits) == reference(
            solver, manager, names, limits, rows
        )


def test_acm_margin_and_aba(cell):
    _env, solver, manager, names, limits, q = cell
    original = manager.getContactAllowedValidator()
    initial_margin = manager.getCollisionMarginData()
    try:
        manager.setCollisionMarginData(CollisionMarginData(0.001))
        a = q.copy()
        b = a.copy()
        positioner = names.index("positioner_joint_1")
        b[positioner] = min(limits[positioner, 1], max(limits[positioner, 0], a[positioner] + 0.2))
        assert a[positioner] != b[positioner]
        rows = np.array([a, b, a])
        assert check_batched_state_rows(solver, manager, names, rows, limits) == reference(
            solver, manager, names, limits, rows
        )
        first = reference(solver, manager, names, limits, q[None, :])
        acm = AllowedCollisionMatrix()
        acm.addAllowedCollision(first[2], first[3], "unit test")
        manager.setContactAllowedValidator(ACMContactAllowedValidator(acm))
        assert check_batched_state_rows(solver, manager, names, rows, limits) == reference(
            solver, manager, names, limits, rows
        )
    finally:
        manager.setContactAllowedValidator(original)
        manager.setCollisionMarginData(initial_margin)


def test_invalid_input_and_unknown_validator_rejected(cell):
    _env, solver, manager, names, limits, q = cell
    for rows, bounds, ids in (
        (q[None, :-1], limits, names),
        (np.full((1, len(names)), np.nan), limits, names),
        (np.empty((0, len(names))), limits, names),
        (q[None, :], limits[:-1], names),
        (q[None, :], limits, (*names[:-1], names[0])),
        (q[None, :], limits, (*names[:-1], "unknown_joint")),
    ):
        with pytest.raises(InvalidBatch):
            check_batched_state_rows(solver, manager, ids, rows, bounds)
    original = manager.getContactAllowedValidator()
    manager.setContactAllowedValidator(CombinedContactAllowedValidator())
    try:
        with pytest.raises(UnsupportedBatchContext, match="validator"):
            check_batched_state_rows(solver, manager, names, q[None, :], limits)
    finally:
        manager.setContactAllowedValidator(original)


def test_independent_checker_threads_and_gil(cell):
    env, solver, manager, names, limits, q = cell
    # Each thread has a distinct mutable manager and solver; no shared checker.
    clean_manager = env.getDiscreteContactManager()
    clean_solver = env.getStateSolver()
    rows = np.repeat(q[None, :], THREAD_PROGRESS_ROWS, axis=0)
    expected = reference(clean_solver, clean_manager, names, limits, rows)
    assert expected[0] == "clear"

    def run():
        return check_batched_state_rows(
            env.getStateSolver(), env.getDiscreteContactManager(), names, rows, limits
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert [future.result() for future in (pool.submit(run), pool.submit(run))] == [
            expected,
            expected,
        ]
    started, stop = Event(), Event()
    counter = [0]

    def advance():
        started.set()
        while not stop.is_set():
            counter[0] += 1

    thread = Thread(target=advance)
    thread.start()
    started.wait()
    before = counter[0]
    try:
        assert (
            check_batched_state_rows(clean_solver, clean_manager, names, rows, limits) == expected
        )
    finally:
        after = counter[0]
        stop.set()
        thread.join()
    assert after > before

    # A second Python thread changes the input only after the native loop has
    # released the GIL. The native function must have copied the input first.
    done = Event()
    mutable_rows = rows.copy()

    def mutate_input():
        sleep(0.01)
        mutable_rows[:] = np.nan
        done.set()

    thread = Thread(target=mutate_input)
    thread.start()
    try:
        assert (
            check_batched_state_rows(clean_solver, clean_manager, names, mutable_rows, limits)
            == expected
        )
    finally:
        thread.join()
    assert done.is_set()


def test_both_import_orders_in_clean_python():
    for first, second in (
        ("tesseract_collision", "tesseract_state_solver"),
        ("tesseract_state_solver", "tesseract_collision"),
    ):
        code = (
            f"import tesseract_robotics.{first}; import tesseract_robotics.{second}; "
            "from tesseract_robotics.tesseract_collision import check_batched_state_rows"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
