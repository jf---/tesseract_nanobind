# Ruckig trajectories

**An owned Ruckig trajectory can be evaluated at a modeled time. The actual curves captured from Tesseract's waypoint smoother have their own native durations; they do not automatically describe the smoother's waypoint clock.** These APIs require the accompanying Tesseract Planning Ruckig capture and retiming changes.

## Direct point-to-point trajectory

`RuckigTrajectory.solve` requires position, velocity and acceleration at both endpoints, positive symmetric per-axis velocity, acceleration and jerk bounds, and a minimum duration in seconds. It returns an owned trajectory; `evaluate(time)` returns copies of position, velocity, acceleration and jerk. Invalid input, solver failure and an out-of-range time raise `RuckigInputError`, `RuckigSolveError` and `RuckigTimeError` respectively. The caller supplies units consistently, and must establish whether the limits reflect the real cell. This generates a new point-to-point curve; it does not preserve a previously planned intermediate path.

```python
from tesseract_robotics.tesseract_time_parameterization import RuckigTrajectory

trajectory = RuckigTrajectory.solve(
    start_position=[0.0], target_position=[0.5],
    start_velocity=[0.0], target_velocity=[0.0],
    start_acceleration=[0.0], target_acceleration=[0.0],
    max_velocity=[2.0], max_acceleration=[4.0], max_jerk=[8.0],
    minimum_duration=1.0,
)
position, velocity, acceleration, jerk = trajectory.evaluate(trajectory.duration / 2)
```

These values are illustrative, not robot limits. Input vectors are copied before native calculation releases the GIL. Evaluation returns copied state vectors.

## Existing smoother and captured segments

`RuckigTrajectorySmoothing.compute` retains its existing behavior. Its `compute_with_trajectories(program, env, profiles)` operation explicitly runs that smoother and returns the actual native trajectories from its successful attempt. Each segment exposes its indices, waypoint interval, native `duration`, `update_time`, `update_state`, `evaluate(time)` and `evaluate_update()`. The program may be changed, including on failure. Do not mutate the program, environment or profiles concurrently during this call.

The smoother uses an average update timestep. Its update can evaluate beyond a native segment's duration; `evaluate_update()` reports that actual native update and does not claim it was written as a waypoint or executed by a controller. Ordinary `evaluate()` is restricted to the native duration. The native segment duration, update time and waypoint interval can differ.

`compute_checked_trajectories(program, env, profiles)` instead operates on a copy. It rejects clock or derivative changes, native durations that do not match their corresponding waypoint intervals, and endpoint-state mismatches. `RuckigSegmentTimingError` identifies the failing segment and reports native and waypoint durations. This checks segment timing alignment; it does not certify collision, process constraints, or controller execution.

The reason for the distinction is observable: a synthetic program moving joint 1 by 0.2 rad in a 0.001 s interval, followed by a 9.999 s interval, returned success from the existing smoother despite a 2 rad/s requested maximum. Its first interval alone requires 200 rad/s mean velocity. The checked operation rejects it. These are modeled test inputs, not cell measurements.

## Explicit free-space retiming

`retime_freespace(program, position_lower, position_upper, max_velocity, max_acceleration, max_jerk)` returns a new program and owned `RuckigRetimedSegment` objects. It requires finite explicit bounds and state waypoints with matching joint order. Only FREESPACE moves are accepted: LINEAR/CIRCULAR motion and embedded event instructions are refused. Each input interval is a minimum; returned waypoint times align with the actual retained Ruckig segment durations. Waypoint position, velocity and acceleration are retained and the input program is unchanged. Native position extrema are checked against the supplied joint bounds.

Each returned segment carries `start_index`, `end_index`, `original_duration`, `start_time`, `end_time`, `duration` and `evaluate(time)`. `RuckigFreespaceError` identifies unsupported or infeasible input. No source path or collision geometry is preserved by this point-to-point construction: check its actual curves independently before use. Probe, weld and fixture process timing are outside this operation. Acceleration and jerk limits that are not sourced must be identified as modeled, never presented as robot limits.
