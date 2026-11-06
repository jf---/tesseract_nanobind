"""Compare ordered collision-only rows against scalar checks in the actual cell."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from candidate_penetration import no_penetration_first
from fresh_product_scene import initialize_scene, snapshot
from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

from tesseract_robotics.tesseract_collision import (
    classify_collision_state_rows_first,
)
from tesseract_robotics.tesseract_common import ACMContactAllowedValidator, AllowedCollisionMatrix


class CollisionOnlyParityError(RuntimeError):
    """Native collision decisions differ from the scalar cell checks."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "captured", "witness", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    scene = initialize_scene(args.station, args.tour_index, args.output)
    setup_s = perf_counter() - started
    initial = snapshot(scene)
    env, _, _, _, names, *_ = scene["inputs"]
    checker = StateChecker.build(env, names, {}, PairMargins.uniform(Meters(0)))
    entries = json.loads(args.captured.read_text())["entries"]
    rows_total = rejected = 0
    for entry in entries:
        fixture = np.asarray(entry["fixture"], dtype=float)
        seven = np.asarray([item["state"] for item in entry["checks"]], dtype=float)
        rows = np.column_stack((seven, np.tile(fixture, (len(seven), 1))))
        expected = [item["clear"] for item in entry["checks"]]
        scalar = [no_penetration_first(checker, row) for row in rows]
        checker._select_acm()
        checker.manager.setCollisionMarginData(checker.margins.margin_data())
        native = classify_collision_state_rows_first(checker.solver, checker.manager, names, rows)
        if scalar != expected or [value == "clear" for value in native] != expected:
            raise CollisionOnlyParityError(f"ordered row parity failed: {entry['label']}")
        rows_total += len(rows)
        rejected += expected.count(False)

    witness = json.loads(args.witness.read_text())
    q = np.asarray(witness["state"], dtype=float)
    witness_checker = StateChecker.build(env, names, {}, PairMargins.uniform(Meters(0.05)))
    first = witness_checker.first_contact(q)
    contacts = witness_checker.contacts(q)
    if (
        first is None
        or float(first.distance_m) < 0
        or not any(float(c.distance_m) < 0 for c in contacts)
    ):
        raise CollisionOnlyParityError("positive-FIRST/negative-ALL witness changed")
    manager = witness_checker.manager
    witness_checker._select_acm()
    manager.setCollisionMarginData(witness_checker.margins.margin_data())
    original = manager.getContactAllowedValidator()
    try:
        before = classify_collision_state_rows_first(
            witness_checker.solver, manager, names, q[None, :]
        )
        acm = AllowedCollisionMatrix(env.getAllowedCollisionMatrix().getAllAllowedCollisions())
        for contact in contacts:
            if float(contact.distance_m) < 0:
                acm.addAllowedCollision(*map(str, contact.links), "collision-only parity")
        manager.setContactAllowedValidator(ACMContactAllowedValidator(acm))
        overridden = classify_collision_state_rows_first(
            witness_checker.solver, manager, names, q[None, :]
        )
        manager.setContactAllowedValidator(original)
        restored = classify_collision_state_rows_first(
            witness_checker.solver, manager, names, q[None, :]
        )
    finally:
        manager.setContactAllowedValidator(original)
    if (before, overridden, restored) != (["sampled_collision"], ["clear"], ["sampled_collision"]):
        raise CollisionOnlyParityError(
            f"ACM witness override or restoration differs: {before}, {overridden}, {restored}"
        )
    if snapshot(scene) != initial:
        raise CollisionOnlyParityError("scene was not restored")
    result = {
        "rows": rows_total,
        "rejected": rejected,
        "fixtures": [entry["fixture"] for entry in entries],
        "positive_first_m": float(first.distance_m),
        "negative_all_m": min(float(c.distance_m) for c in contacts),
        "witness_verdicts": [before, overridden, restored],
        "fixture_scene_restored": True,
        "setup_s": setup_s,
        "scope": "actual generated candidate rows, collision only, finite sampled checks",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
