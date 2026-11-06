#include "tesseract_nb.h"

#include <array>
#include <cmath>
#include <memory>
#include <ruckig/ruckig.hpp>
#include <stdexcept>
#include <string>
#include <tesseract/command_language/composite_instruction.h>
#include <tesseract/common/profile_dictionary.h>
#include <tesseract/environment/environment.h>
#include <tesseract/time_parameterization/ruckig/ruckig_freespace_retiming.h>
#include <tesseract/time_parameterization/ruckig/ruckig_trajectory_smoothing.h>
#include <vector>

namespace {
struct RuckigInputError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};
struct RuckigTimeError : std::out_of_range {
  using std::out_of_range::out_of_range;
};
struct RuckigSolveError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
using Values = std::vector<double>;
using State = std::array<Values, 4>;

class RuckigTrajectory {
public:
  explicit RuckigTrajectory(std::size_t dof) : trajectory_(dof), dof_(dof) {}

  static std::unique_ptr<RuckigTrajectory>
  solve(Values start_position, Values target_position, Values start_velocity,
        Values target_velocity, Values start_acceleration,
        Values target_acceleration, Values max_velocity,
        Values max_acceleration, Values max_jerk, double minimum_duration) {
    const std::size_t dof = start_position.size();
    const std::array<const Values *, 9> arrays = {
        &start_position,  &target_position,    &start_velocity,
        &target_velocity, &start_acceleration, &target_acceleration,
        &max_velocity,    &max_acceleration,   &max_jerk};
    if (dof == 0 || !std::isfinite(minimum_duration) || minimum_duration < 0.0)
      throw RuckigInputError(
          "expected nonempty joints and finite nonnegative minimum duration");
    for (const Values *values : arrays) {
      if (values->size() != dof)
        throw RuckigInputError(
            "every state and limit must have one value per joint");
      for (double value : *values)
        if (!std::isfinite(value))
          throw RuckigInputError("state and limit values must be finite");
    }
    for (const Values *limits : {&max_velocity, &max_acceleration, &max_jerk})
      for (double limit : *limits)
        if (limit <= 0.0)
          throw RuckigInputError(
              "all symmetric limits must be strictly positive");
    ruckig::InputParameter<ruckig::DynamicDOFs> input(dof);
    input.current_position = std::move(start_position);
    input.target_position = std::move(target_position);
    input.current_velocity = std::move(start_velocity);
    input.target_velocity = std::move(target_velocity);
    input.current_acceleration = std::move(start_acceleration);
    input.target_acceleration = std::move(target_acceleration);
    input.max_velocity = std::move(max_velocity);
    input.max_acceleration = std::move(max_acceleration);
    input.max_jerk = std::move(max_jerk);
    input.minimum_duration = minimum_duration;
    auto result = std::make_unique<RuckigTrajectory>(dof);
    try {
      ruckig::Ruckig<ruckig::DynamicDOFs> generator(dof);
      const auto status = generator.calculate(input, result->trajectory_);
      if (static_cast<int>(status) < 0)
        throw RuckigSolveError("Ruckig calculation returned status " +
                               std::to_string(static_cast<int>(status)));
    } catch (const ruckig::RuckigError &error) {
      throw RuckigSolveError(error.what());
    }
    return result;
  }

  double duration() const { return trajectory_.get_duration(); }

  State evaluate(double time) const {
    if (!std::isfinite(time) || time < 0.0 || time > duration())
      throw RuckigTimeError(
          "evaluation time must be finite and within trajectory duration");
    State state{Values(dof_), Values(dof_), Values(dof_), Values(dof_)};
    std::size_t section = 0;
    trajectory_.at_time(time, state[0], state[1], state[2], state[3], section);
    return state;
  }

private:
  ruckig::Trajectory<ruckig::DynamicDOFs> trajectory_;
  std::size_t dof_;
};
} // namespace

void bind_ruckig_trajectory(nb::module_ &m) {
  nb::exception<RuckigInputError>(m, "RuckigInputError", PyExc_ValueError);
  nb::exception<RuckigTimeError>(m, "RuckigTimeError", PyExc_ValueError);
  nb::exception<RuckigSolveError>(m, "RuckigSolveError", PyExc_RuntimeError);
  nb::exception<tesseract::time_parameterization::RuckigSegmentTimingError>(
      m, "RuckigSegmentTimingError", PyExc_RuntimeError);
  nb::exception<tesseract::time_parameterization::RuckigFreespaceError>(
      m, "RuckigFreespaceError", PyExc_RuntimeError);
  nb::class_<RuckigTrajectory>(m, "RuckigTrajectory",
                               "Owned direct point-to-point Ruckig trajectory; "
                               "not the Tesseract waypoint smoother's output.")
      .def_static("solve", &RuckigTrajectory::solve, "start_position"_a,
                  "target_position"_a, "start_velocity"_a, "target_velocity"_a,
                  "start_acceleration"_a, "target_acceleration"_a,
                  "max_velocity"_a, "max_acceleration"_a, "max_jerk"_a,
                  "minimum_duration"_a,
                  nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("duration", &RuckigTrajectory::duration)
      .def("evaluate", &RuckigTrajectory::evaluate, "time"_a,
           nb::call_guard<nb::gil_scoped_release>(),
           "Return position, velocity, acceleration and jerk at time in "
           "seconds.");

  using RetimedSegment = tesseract::time_parameterization::RuckigRetimedSegment;
  nb::class_<RetimedSegment>(m, "RuckigRetimedSegment")
      .def_ro("start_index", &RetimedSegment::start_index)
      .def_ro("end_index", &RetimedSegment::end_index)
      .def_ro("original_duration", &RetimedSegment::original_duration)
      .def_ro("start_time", &RetimedSegment::start_time)
      .def_ro("end_time", &RetimedSegment::end_time)
      .def_prop_ro("duration",
                   [](const RetimedSegment &segment) {
                     return segment.trajectory.get_duration();
                   })
      .def(
          "evaluate",
          [](const RetimedSegment &segment, double time) {
            if (!std::isfinite(time) || time < 0.0 ||
                time > segment.trajectory.get_duration())
              throw RuckigTimeError("evaluation time must be finite and within "
                                    "native trajectory duration");
            const auto dof = segment.trajectory.degrees_of_freedom;
            State state{Values(dof), Values(dof), Values(dof), Values(dof)};
            std::size_t section = 0;
            segment.trajectory.at_time(time, state[0], state[1], state[2],
                                       state[3], section);
            return state;
          },
          "time"_a, nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "retime_freespace",
      [](const tesseract::command_language::CompositeInstruction &program,
         Values position_lower, Values position_upper, Values max_velocity,
         Values max_acceleration, Values max_jerk) {
        // Copy the program and all converted vectors before releasing the GIL.
        auto input = program;
        nb::gil_scoped_release release;
        return tesseract::time_parameterization::retimeFreespace(
            input, position_lower, position_upper, max_velocity,
            max_acceleration, max_jerk);
      },
      "program"_a, "position_lower"_a, "position_upper"_a, "max_velocity"_a,
      "max_acceleration"_a, "max_jerk"_a,
      "Return a new free-space-only program and actual owned Ruckig segments. "
      "Explicit limits are required; source q/v/a are preserved and original "
      "intervals are minima. This does not check collision or process "
      "semantics.");
}

void bind_ruckig_smoother_segments(
    nb::module_ &m,
    nb::class_<tesseract::time_parameterization::RuckigTrajectorySmoothing,
               tesseract::time_parameterization::TimeParameterization>
        &smoother) {
  using Segment = tesseract::time_parameterization::RuckigSmoothingSegment;
  auto evaluate_native = [](const Segment &segment, double time) {
    const auto dof = segment.update_position.size();
    State state{Values(dof), Values(dof), Values(dof), Values(dof)};
    std::size_t section = 0;
    segment.trajectory.at_time(time, state[0], state[1], state[2], state[3],
                               section);
    return state;
  };
  nb::class_<Segment>(m, "RuckigSmoothingSegment",
                      "An owned native trajectory and exact update from the "
                      "smoother's successful attempt.")
      .def_ro("start_index", &Segment::start_index)
      .def_ro("end_index", &Segment::end_index)
      .def_ro("waypoint_start_time", &Segment::waypoint_start_time)
      .def_ro("waypoint_end_time", &Segment::waypoint_end_time)
      .def_ro("update_time", &Segment::update_time)
      .def_prop_ro("duration",
                   [](const Segment &segment) {
                     return segment.trajectory.get_duration();
                   })
      .def_prop_ro("update_state",
                   [](const Segment &segment) {
                     return State{
                         segment.update_position, segment.update_velocity,
                         segment.update_acceleration, segment.update_jerk};
                   })
      .def(
          "evaluate",
          [evaluate_native](const Segment &segment, double time) {
            if (!std::isfinite(time) || time < 0.0 ||
                time > segment.trajectory.get_duration())
              throw RuckigTimeError("evaluation time must be finite and within "
                                    "native trajectory duration");
            return evaluate_native(segment, time);
          },
          "time"_a, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "evaluate_update",
          [evaluate_native](const Segment &segment) {
            return evaluate_native(segment, segment.update_time);
          },
          nb::call_guard<nb::gil_scoped_release>(),
          "Evaluate at the smoother update time, which can exceed the native "
          "duration.");
  smoother.def(
      "compute_with_trajectories",
      [](const tesseract::time_parameterization::RuckigTrajectorySmoothing
             &self,
         tesseract::command_language::CompositeInstruction &program,
         const tesseract::environment::Environment &environment,
         const tesseract::common::ProfileDictionary &profiles) {
        std::vector<Segment> segments;
        if (!self.computeWithTrajectories(program, environment, profiles,
                                          segments))
          throw RuckigSolveError("smoother failed or its input clock is "
                                 "unsupported; program may have changed");
        return segments;
      },
      "program"_a, "env"_a, "profiles"_a,
      nb::call_guard<nb::gil_scoped_release>(),
      "Capture the final successful attempt's exact native updates. Waypoint "
      "intervals, native durations and update times are distinct; they do not "
      "define an assembled clock. The program may be modified even on "
      "failure.");
  smoother.def(
      "compute_checked_trajectories",
      [](const tesseract::time_parameterization::RuckigTrajectorySmoothing
             &self,
         const tesseract::command_language::CompositeInstruction &program,
         const tesseract::environment::Environment &environment,
         const tesseract::common::ProfileDictionary &profiles) {
        // Copy before releasing the GIL; the checked operation never mutates
        // the caller's program.
        auto input = program;
        nb::gil_scoped_release release;
        std::vector<Segment> segments;
        if (!self.computeCheckedWithTrajectories(input, environment, profiles,
                                                 segments))
          throw RuckigSolveError(
              "smoother failed or its input clock is unsupported");
        return segments;
      },
      "program"_a, "env"_a, "profiles"_a,
      "Check a copy without retiming or changing derivatives. Reject a native "
      "duration that does not match its waypoint interval; no implicit hold or "
      "extrapolation is assumed. This checks segment timing alignment, not "
      "collision, process or controller acceptance.");
}
