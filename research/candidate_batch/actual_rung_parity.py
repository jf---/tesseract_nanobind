"""Verify and time captured generated rungs in a single owned actual-cell scene."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from candidate_chunks import classify_candidate_chunks
from candidate_penetration import no_penetration_first
from fresh_product_scene import initialize_scene, snapshot
from tesseract_qt_py.core.retarget.certify import StateChecker
from tesseract_qt_py.core.retarget.pair_margins import PairMargins
from tesseract_qt_py.core.types import Meters

from tesseract_robotics.tesseract_collision import classify_candidate_state_rows

ROOT = Path("/Users/jelle/Code/CADCAM/tesseract_qt_robodk/.worktrees/r62-smooth-windows-main")
ORIGINAL = ROOT.parent / "r62-smooth-windows"
STATION = (
    ORIGINAL
    / ".superpowers/r62-master/tmp/.claude/logs/p2/cache/verbotics/31b3c0640a7963e90688e6b979eeb4c5370bbe303aab387b4f46844edf51e4e8"
)
TOURS = (
    ORIGINAL
    / "docs/superpowers/reviews/2026-10-23-replace-p2-resolve-routes/probes/raster_tour/budgeted/program_tours_timed/program_tours.json"
)
INPUT = ROOT / ".superpowers/candidate-rung-inputs/actual-tour00/results.json"
OUTPUT = Path(__file__).parent / "cell-output"
CHUNK_SIZE = 64
REPEATS = 3

OUTPUT.mkdir(exist_ok=True)
start = perf_counter()
scene = initialize_scene(STATION, TOURS, OUTPUT)
setup = perf_counter() - start
initial = snapshot(scene)
env, _, _, _, names, *_ = scene["inputs"]
limits = np.asarray(
    env.getJointGroup("candidate_rung_limits", list(names)).getLimits().joint_limits
)
entries = json.loads(INPUT.read_text())["entries"]
full = StateChecker.build(env, names, {}, PairMargins.uniform(Meters(0)))


def evaluate(entry, measure):
    fixture = entry["fixture"]
    held = dict(zip(names[-2:], fixture, strict=True))
    scalar = StateChecker.build(env, names[:7], held, PairMargins.uniform(Meters(0)))
    seven = np.asarray([item["state"] for item in entry["checks"]], dtype=float)
    rows = np.column_stack([seven, np.tile(fixture, (len(seven), 1))])
    expected = [item["clear"] for item in entry["checks"]]

    def scalar_call():
        return [no_penetration_first(scalar, row) for row in seven]

    def native_call():
        def check(chunk):
            full._select_acm()
            full.manager.setCollisionMarginData(full.margins.margin_data())
            return classify_candidate_state_rows(
                full.solver, full.manager, full.joint_names, chunk, limits
            )

        return [
            result == "clear"
            for result in classify_candidate_chunks(
                rows, check, lambda: None, chunk_size=CHUNK_SIZE
            )
        ]

    scalar_result = scalar_call()
    native_result = native_call()
    if scalar_result != expected or native_result != expected:
        raise RuntimeError(f"ordered candidate parity failed for {entry['label']}")
    times = {"scalar_s": [], "native_s": []}
    for repeat in range(REPEATS if measure else 0):
        order = [("scalar_s", scalar_call), ("native_s", native_call)]
        if repeat % 2:
            order.reverse()
        for label, call in order:
            started = perf_counter()
            verdicts = call()
            times[label].append(perf_counter() - started)
            if verdicts != expected:
                raise RuntimeError("timed candidate verdict differs")
    return {
        "fixture": fixture,
        "label": entry["label"],
        "rows": len(rows),
        "accepted": sum(expected),
        "times": times,
    }


for entry in entries:
    evaluate(entry, False)
results = [evaluate(entry, True) for entry in entries]
if snapshot(scene) != initial:
    raise RuntimeError("scene did not remain unchanged")
report = {
    "scope": "Same saved generated candidate contact checks, one retained scene; no IK, generation, planning or tour throughput included.",
    "setup_s": setup,
    "chunk_size": CHUNK_SIZE,
    "repeats": REPEATS,
    "results": results,
    "scene_restored": True,
}
(OUTPUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
