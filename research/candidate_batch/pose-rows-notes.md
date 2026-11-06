# OFKT workpiece pose rows

`workpiece_pose_residual_rows` is an additive research API. It returns ordered
metre and radian residuals for supplied workpiece-frame target matrices. For
each row it computes the workpiece transform using only the named held joints,
then computes TCP FK with all supplied joints, exactly as the scalar callback
does. Target interpolation remains with the caller. Inputs are copied before
native work releases the GIL; the caller must own the OFKT solver exclusively
and keep topology fixed. Unsupported solvers and malformed input fail loudly.

The API performs no collision or deadline checks. Callers using chunks must
check deadlines before and after native work and discard expired results;
exceptions can arrive at chunk rather than per-row boundaries. It is not yet
connected to the planner. Real-cell parity evidence covers 1,617 generated
process/witness rows, A→B→A fixtures and cases around both process thresholds,
zero and near-pi rotation. This is sampled correctness evidence, not a
performance result or physical validation. The actual-cell check takes the
station, tour-index, candidate, witness and output paths as explicit arguments.
