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

## macOS wheel experiment

On 2026-11-06, an isolated macOS arm64 build produced a research wheel for
macOS 15.0 and later. The native libraries and extensions were built for
15.0; all 90 libraries in the explicit manifest reported minimum versions no
higher than 15.0. `packaging/macos-piqp-libraries.txt` records the library
install names and their sources: `stage/` for the matched local builds and
`pixi/` for the Pixi dependency environment. qpOASES 3.2.2 was built from its
upstream release with install RPATH disabled because the published conda
library embedded a build-machine path.

The opt-in `TESSERACT_MACOS_LIBRARY_LIST` CMake setting accepts a file of
absolute library paths, rejects missing files and duplicate install names,
copies symlink targets under the required names, and uses loader-relative
extension RPATHs. The caller resolves each `stage/` or `pixi/` entry in the
source manifest to its own staged or Pixi `lib` directory. Without this
setting, the existing CMake behavior is unchanged. This path assembles the
wheel directly; it does not invoke delocate's signing or repair command.

The wheel contained all 90 named libraries and 472 files, all represented in
its wheel record. An inspection of 113 Mach-O files found no absolute external
RPATH or non-system absolute dependency. In a separate Pixi consumer with
both DYLD library path variables unset, 32 focused tests passed, including
PIQP, callbacks, collision checks and compatibility APIs. A real-cell Tour 20
also passed 10,260 sampled checks and a separate Python recheck; Bullet and
OPW plugin factories and the corrected native libraries loaded from the wheel.
The tour used direct transits, so it did not exercise a routed PIQP solve.

The real-cell runner and `robot_parser` were loaded from local source, and the
test host ran macOS 15.3.2; this is not a fully installed-application test or
a test on macOS 15.0. It does not establish controller execution, physical
collision freedom or performance across tours. Other macOS versions and
architectures, and the third-party license inventory for distribution, remain
unverified. The research wheel is not a release artifact.

## PIQP timing probe

`PIQPTimedSolver` is an opt-in research subclass. It delegates each call to the
existing PIQP implementation, enables PIQP's timing counters and records QP
wall, setup, bounds-update and solve times. A rejected call is recorded as a
failure with native status and timings unavailable: the underlying solver may
reject crossed bounds before entering PIQP, so its previous result must not be
reported as the failed call's timing. The ordinary `PIQPSolver` is unchanged.

On two generated detours in one retained scene, both the ordinary and timed
soft-to-hard solves converged to identical joint states and passed independent
sampled checks. Across their four timed stages, wall time was 0.731 s: term
creation 0.241 s, problem assembly/setup 0.025 s and SQP 0.464 s. The 88 QP
calls used 0.073 s wall, including 0.010 s of PIQP setup; 73 calls set up and
15 updated bounds. These are two bounded examples, not an all-tour benchmark.
SQP work outside the QP calls was not separated into collision, derivatives
and other work in this probe. Scene setup, 4.83 s, was measured separately.

`TimedTrajOptQPProblem` is a separate opt-in research problem that measures
existing SQP entry points: variable updates, convexification, exact costs and
constraints, and convex costs and constraints. These timings are inclusive;
convexification includes derivative assembly, while variable updates visit all
components and can include collision work. Neither timer isolates collision by
itself.

A generated Tour 18 detour matched a retained route's endpoints and fixture.
For one soft-to-hard replay, baseline and timed variants produced identical
joint states and the same independent-check result; both SQP stages converged,
while the existing sampled-check policy rejected the candidate. The timed
stages spent 1.940 s and 0.539 s in SQP. Their variable updates took 1.048 s
and 0.135 s, convexification 0.155 s and 0.050 s, and QP calls 0.732 s and
0.354 s. These are inclusive measurements from one run under heavy unrelated
CPU contention, not a clean performance comparison or an all-tour result.

## Combined Ruckig and GIL wheel experiment

A second isolated macOS arm64 research wheel combines the tested FK/IK input
ownership and OFKT concurrency changes with the owned Ruckig trajectory binding.
The nanobind source branch is `codex/piqp-ruckig-gil` at `6ad58f0`, based on
`684f248`; its matched Planning Ruckig source is `16c50aa`. The native Ruckig
library was rebuilt for macOS 15.0 into a separate staging prefix; the earlier
evidence wheel and source installations were not changed.

In a separate Pixi consumer with `DYLD_LIBRARY_PATH` and
`DYLD_FALLBACK_LIBRARY_PATH` unset, 95 focused and reference tests passed,
covering PIQP, Ruckig, FK/IK ownership and the OFKT-specific entry point. A
wheel inspection found 472 recorded files and 113 Mach-O files with no external
absolute RPATH or non-system absolute dependency; each reported a minimum macOS
version no later than 15.0. The blocking IK test helper is a separately built
research module; the test process preloads the wheel's kinematics module to
prevent the helper's build-time RPATH from selecting a second native library.

The R62 native-cell smoke loaded Bullet and OFKT from the wheel, reproduced
fixture A–B–A contacts, matched all 62 OFKT link transforms across its two
state APIs, and evaluated a constructed seven-axis Ruckig motion through the
cell's joint group. It used local source for the kernel artifact locator and
research checker. The older application research loader expects an artifact
locator API that the current standalone parser no longer exports, so that
loader is not established as compatible by this component smoke. The synthetic
motion is not a planned tour, continuous collision proof, or controller result.
The wheel is a local research artifact; other macOS versions, architectures,
third-party licensing for distribution, and physical execution remain unchecked.
