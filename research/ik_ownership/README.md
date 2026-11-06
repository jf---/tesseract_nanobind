# Kinematics and OFKT ownership probe

This isolated development build leaves the packaged evidence wheel unchanged.
`JointGroup.calcFwdKin` copies its joint vector while holding the GIL, validates
its size, and releases the GIL only for native FK. Inverse-kinematics bindings
copy joint seeds and mutable pose containers before the native call. Their
native solver has no Python trampoline; native exceptions are checked by a
blocking test solver.

`OFKTStateSolver.getStateConcurrent` is an explicit additive map entry point.
It copies and validates inputs before release. The inspected OFKT implementation
synchronizes state reads and updates; topology must stay fixed across validation
and the call. It is not available on KDL and the existing `getState` overloads
remain untouched. Use independent scene and group owners; do not concurrently
mutate a group or solver whose native implementation does not support it.

Build `probe.cpp` with the adjacent CMake project, then include the resulting
module and this isolated wheel overlay on `PYTHONPATH` when running the focused
pytest files. The blocked native solver makes Python mutate the seed while an IK
call is in progress. The unfixed binding returned the mutated value; the fixed
binding returns the starting value. This does not establish the safety of Python
callbacks or third-party solver plugins, nor prove the exact provenance of the
packaged native state solver.
