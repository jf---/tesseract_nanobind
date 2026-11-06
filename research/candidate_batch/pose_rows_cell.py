"""Compare native pose residual rows against scalar R62 fixture-frame FK."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from fresh_product_scene import initialize_scene, snapshot
from native_weld_candidates import build_native_callbacks
from process_interval_validation import INTERVAL_SAMPLES, ORIENTATION_SCREEN_RAD, POSITION_SCREEN_M
from sampled_weld_graph import _interpolate
from tesseract_qt_py.core.types import LinkName
from weld_only_inputs import FixtureState, WorkpiecePose

from tesseract_robotics.tesseract_state_solver import workpiece_pose_residual_rows

# Double-precision real-cell FK/residual parity; 1 pm and 1 prad bound arithmetic differences.
RESIDUAL_PARITY_ATOL = 1e-12
# One micrometre and one microradian around each existing coarse process threshold.
THRESHOLD_OFFSET_M = 1e-6
THRESHOLD_OFFSET_RAD = 1e-6
# Bounded work uses the same deadline granularity as the process collision prototype.
CHUNK_SIZE = 64


class PoseRowsParityError(RuntimeError):
    """The native residuals or threshold decisions differ from the scalar path."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("station", "tour-index", "candidate", "witness", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 0:
        raise PoseRowsParityError("repeat count must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    scene = initialize_scene(args.station, args.tour_index, args.output)
    initial = snapshot(scene)
    env, _, _, _, names, opw, *_ = scene["inputs"]
    solver = env.getStateSolver()
    saved = json.loads(args.candidate.read_text())
    welds = []
    for index, (kind, edge) in enumerate(
        zip(saved["edge_kinds"], saved["result"]["edges"], strict=True)
    ):
        if kind != "weld":
            continue
        states = np.asarray(edge["states"], dtype=float)
        if states.ndim != 2 or states.shape[1] != 9 or not np.all(states[:, 7:] == states[0, 7:]):
            raise PoseRowsParityError("expected fixed-fixture nine-axis weld")
        if not welds or not np.array_equal(states[0, 7:], welds[0][1][0, 7:]):
            welds.append((index, states))
        if len(welds) == 2:
            break
    if len(welds) != 2:
        raise PoseRowsParityError("two fixture positions are required")
    welds.append(welds[0])
    witness = np.asarray(json.loads(args.witness.read_text())["state"], dtype=float)
    if witness.shape != (9,) or not np.isfinite(witness).all():
        raise PoseRowsParityError("finite nine-axis witness required")
    welds.append(("penetration-witness", np.stack((witness, witness))))

    def pose(full):
        state = env.getState(dict(zip(names, map(float, full), strict=True)))
        workpiece = np.asarray(state.link_transforms["workpiece_link"].matrix)
        tcp = np.asarray(state.link_transforms["nozzle_link"].matrix)
        return WorkpiecePose.build(np.linalg.solve(workpiece, tcp))

    records = []
    all_ok = True
    for index, states in welds:
        fixture = FixtureState(*map(float, states[0, 7:]))
        bead = SimpleNamespace(fixture=fixture, workpiece_frame=LinkName("workpiece_link"))
        _, _, error, _ = build_native_callbacks(
            env, opw, tuple(names[:7]), "nozzle_link", bead, use_first_contact=True
        )
        start, end = pose(states[0]), pose(states[-1])
        rows = []
        poses = []
        for i, (a, b) in enumerate(zip(states, states[1:])):
            for u in np.linspace(0.0, 1.0, INTERVAL_SAMPLES):
                rows.append(a * (1.0 - u) + b * u)
                poses.append(_interpolate(start, end, (i + float(u)) / (len(states) - 1)))
        rows = np.asarray(rows)
        target_matrices = []
        for target in poses:
            matrix = np.eye(4)
            matrix[:3, :3] = target.rotation
            matrix[:3, 3] = target.translation_m
            target_matrices.append(matrix)
        scalar = np.asarray(
            [
                error(tuple(row[:7]), target, fixture, bead.workpiece_frame)
                for row, target in zip(rows, poses, strict=True)
            ]
        )
        native = np.asarray(
            workpiece_pose_residual_rows(
                solver,
                names,
                names[-2:],
                rows,
                np.asarray(target_matrices).reshape(-1, 16),
                "workpiece_link",
                "nozzle_link",
            )
        )
        differences = np.max(np.abs(scalar - native), axis=0)
        scalar_decisions = (scalar[:, 0] <= POSITION_SCREEN_M) & (
            scalar[:, 1] <= ORIENTATION_SCREEN_RAD
        )
        native_decisions = (native[:, 0] <= POSITION_SCREEN_M) & (
            native[:, 1] <= ORIENTATION_SCREEN_RAD
        )
        mismatches = int(np.count_nonzero(scalar_decisions != native_decisions))
        good = bool(np.all(differences <= RESIDUAL_PARITY_ATOL) and mismatches == 0)
        all_ok &= good
        targets_array = np.asarray(target_matrices).reshape(-1, 16)
        repeats = []
        for repeat in range(args.repeats):
            row_timing = {}
            modes = ["scalar", "native"] if repeat % 2 == 0 else ["native", "scalar"]
            for mode in modes:
                began = perf_counter()
                if mode == "scalar":
                    observed = np.asarray(
                        [
                            error(tuple(row[:7]), target, fixture, bead.workpiece_frame)
                            for row, target in zip(rows, poses, strict=True)
                        ]
                    )
                else:
                    observed = np.vstack(
                        [
                            workpiece_pose_residual_rows(
                                solver,
                                names,
                                names[-2:],
                                rows[start : start + CHUNK_SIZE],
                                targets_array[start : start + CHUNK_SIZE],
                                "workpiece_link",
                                "nozzle_link",
                            )
                            for start in range(0, len(rows), CHUNK_SIZE)
                        ]
                    )
                row_timing[mode + "_s"] = perf_counter() - began
                if not np.allclose(observed, scalar, atol=RESIDUAL_PARITY_ATOL, rtol=0):
                    raise PoseRowsParityError(f"timed {mode} residuals differ")
                observed_decisions = (observed[:, 0] <= POSITION_SCREEN_M) & (
                    observed[:, 1] <= ORIENTATION_SCREEN_RAD
                )
                if not np.array_equal(observed_decisions, scalar_decisions):
                    raise PoseRowsParityError(f"timed {mode} decisions differ")
            repeats.append(row_timing)
        records.append(
            {
                "edge": index,
                "fixture": states[0, 7:].tolist(),
                "rows": len(rows),
                "max_position_difference_m": float(differences[0]),
                "max_orientation_difference_rad": float(differences[1]),
                "decision_mismatches": mismatches,
                "matched": good,
                "alternating_repeats": repeats,
            }
        )

    q = welds[0][1][0]
    fixture = FixtureState(*map(float, q[-2:]))
    bead = SimpleNamespace(fixture=fixture, workpiece_frame=LinkName("workpiece_link"))
    _, _, error, _ = build_native_callbacks(
        env, opw, tuple(names[:7]), "nozzle_link", bead, use_first_contact=True
    )
    base_pose = pose(q)
    base = np.eye(4)
    base[:3, :3] = base_pose.rotation
    base[:3, 3] = base_pose.translation_m
    targets = []
    labels = []
    for delta in (
        float(POSITION_SCREEN_M) - THRESHOLD_OFFSET_M,
        float(POSITION_SCREEN_M),
        float(POSITION_SCREEN_M) + THRESHOLD_OFFSET_M,
    ):
        matrix = base.copy()
        matrix[0, 3] += delta
        targets.append(matrix)
        labels.append(f"position_offset_{delta}")
    for angle in (
        0.0,
        float(ORIENTATION_SCREEN_RAD) - THRESHOLD_OFFSET_RAD,
        float(ORIENTATION_SCREEN_RAD),
        float(ORIENTATION_SCREEN_RAD) + THRESHOLD_OFFSET_RAD,
        math.pi - THRESHOLD_OFFSET_RAD,
    ):
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        matrix = base.copy()
        matrix[:3, :3] = base[:3, :3] @ rotation
        targets.append(matrix)
        labels.append(f"rotation_offset_{angle}")
    rows = np.repeat(q[None, :], len(targets), axis=0)
    scalar = np.asarray(
        [
            error(tuple(q[:7]), WorkpiecePose.build(matrix), fixture, bead.workpiece_frame)
            for matrix in targets
        ]
    )
    native = np.asarray(
        workpiece_pose_residual_rows(
            solver,
            names,
            names[-2:],
            rows,
            np.asarray(targets).reshape(-1, 16),
            "workpiece_link",
            "nozzle_link",
        )
    )
    boundary = []
    for label, first, second in zip(labels, scalar, native, strict=True):
        scalar_ok = bool(first[0] <= POSITION_SCREEN_M and first[1] <= ORIENTATION_SCREEN_RAD)
        native_ok = bool(second[0] <= POSITION_SCREEN_M and second[1] <= ORIENTATION_SCREEN_RAD)
        matched = bool(
            np.max(np.abs(first - second)) <= RESIDUAL_PARITY_ATOL and scalar_ok == native_ok
        )
        all_ok &= matched
        boundary.append(
            {
                "label": label,
                "scalar": first.tolist(),
                "native": second.tolist(),
                "scalar_accept": scalar_ok,
                "native_accept": native_ok,
                "matched": matched,
            }
        )
    if snapshot(scene) != initial:
        raise PoseRowsParityError("scene was not restored")
    report = {
        "matched": all_ok,
        "scene_restored": True,
        "chunk_size": CHUNK_SIZE,
        "timing_scope": "precomputed Python-interpolated targets; native copies inputs inside each call; no collision checking or interpolation time",
        "records": records,
        "boundary": boundary,
        "scope": "saved generated process samples plus real-cell threshold and near-pi targets; correctness only",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not all_ok:
        raise PoseRowsParityError("native residual or threshold decisions differ")


if __name__ == "__main__":
    main()
