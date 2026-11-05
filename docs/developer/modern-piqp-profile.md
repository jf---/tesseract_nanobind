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
