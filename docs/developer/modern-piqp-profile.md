# Modern PIQP research build

This isolated build targets the current TrajOpt SQP API and requires its PIQP
backend. Compilation fails if the selected TrajOpt library lacks PIQP.
The selected research solver includes local corrections to SQP step acceptance,
timeouts and terminal status handling; it is not an unmodified upstream build.
It is a separate Python environment and native library set from the 0.35
build. Do not load extensions from the two builds into the same process or
mix their headers and native libraries. The existing 0.35 source, wheel and
tests remain available separately.

The modern `SQPParameters.max_iter` limits convexifications in each penalty
iteration. Rejected trust-region steps do not count towards this limit. It is
not the old `max_iterations` limit: the older parameter uses cumulative QP
calls and is checked before each convexification. There is no alias between
them. Configure the modern parameter explicitly when migrating a planner.

TrajOpt's modern line-constraint method and the accepted bounds for QP
constraint and penalty rows also differ from the 0.35 APIs. Consumers must
test those contracts against this build rather than assume the old behavior.

The batch contact entry point is opt-in. It supports worker-owned OFKT state
solvers and Bullet BVH managers with the native environment or allowed-contact
matrix validators. Other implementations and validators raise an error.
The manager's configured collision margins and allowed contacts remain in
effect. A sampled clear result does not prove continuous collision freedom.

The optional Descartes BGL profile uses lazy Dijkstra and leaves the default
ladder profile unchanged. Its vertex build is restricted to one thread. All
sample, state and edge costs must be finite and nonnegative, including edges
the lazy search never visits. The implementation checks generated vertices and
observed edges and reports invalid observed costs; that check cannot establish
the contract for an arbitrary evaluator's unvisited edges. The supplied scaled
squared-distance evaluator has nonnegative costs for finite inputs.

This is source preparation, not a release wheel or a claim that the complete
application works with the modern build. Installation and consumer results
must be checked against the assembled native libraries and wheel.
