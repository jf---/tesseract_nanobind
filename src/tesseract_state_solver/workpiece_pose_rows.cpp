#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tesseract/scene_graph/scene_state.h>
#include <tesseract/state_solver/ofkt/ofkt_state_solver.h>
#include <tesseract/state_solver/state_solver.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
namespace tc = tesseract::common;
namespace ts = tesseract::scene_graph;

namespace {
class PoseBatchInputError : public std::runtime_error { using std::runtime_error::runtime_error; };
class PoseBatchContextError : public std::runtime_error { using std::runtime_error::runtime_error; };

// Same dimensionless tolerance as WorkpiecePose's double-precision rigid-transform validation.
constexpr double RIGID_TRANSFORM_ATOL = 1e-8;

Eigen::MatrixXd residuals(std::shared_ptr<ts::StateSolver> solver,
                          std::vector<std::string> names,
                          std::vector<std::string> held_names,
                          Eigen::MatrixXd rows,
                          Eigen::MatrixXd targets,
                          std::string workpiece,
                          std::string tcp)
{
  if (!solver)
    throw PoseBatchContextError("an OFKT solver is required");
  const auto& solver_ref = *solver.get();
  if (typeid(solver_ref) != typeid(ts::OFKTStateSolver))
    throw PoseBatchContextError("only an exclusively owned native OFKT solver is supported");
  if (names.empty() || rows.rows() < 1 || rows.cols() != static_cast<Eigen::Index>(names.size()) ||
      targets.rows() != rows.rows() || targets.cols() != 16 || !rows.allFinite() || !targets.allFinite())
    throw PoseBatchInputError("expected finite joint rows and one 4x4 target per row");
  std::set<std::string> unique;
  std::vector<tc::JointId> ids;
  std::vector<Eigen::Index> held_indices;
  const auto available = solver->getActiveJointIds();
  for (const auto& name : names)
  {
    tc::JointId id(name);
    if (name.empty() || !unique.insert(name).second ||
        std::find(available.begin(), available.end(), id) == available.end())
      throw PoseBatchInputError("joint names must be unique, active and known");
    ids.push_back(std::move(id));
  }
  std::set<std::string> unique_held;
  for (const auto& name : held_names)
  {
    auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end() || !unique_held.insert(name).second)
      throw PoseBatchInputError("held joints must be unique members of joint names");
    held_indices.push_back(static_cast<Eigen::Index>(std::distance(names.begin(), it)));
  }
  const tc::LinkId workpiece_id(workpiece), tcp_id(tcp);
  if (workpiece.empty() || tcp.empty() || !solver->hasLinkId(workpiece_id) || !solver->hasLinkId(tcp_id))
    throw PoseBatchInputError("workpiece and TCP links must exist");
  std::vector<Eigen::Matrix4d> owned_targets;
  owned_targets.reserve(static_cast<std::size_t>(targets.rows()));
  for (Eigen::Index i = 0; i < targets.rows(); ++i)
  {
    Eigen::Matrix4d target;
    for (Eigen::Index r = 0; r < 4; ++r)
      for (Eigen::Index c = 0; c < 4; ++c)
        target(r, c) = targets(i, r * 4 + c);
    const Eigen::Matrix3d rotation = target.block<3, 3>(0, 0);
    if ((target.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).cwiseAbs().maxCoeff() > RIGID_TRANSFORM_ATOL ||
        (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > RIGID_TRANSFORM_ATOL ||
        std::abs(rotation.determinant() - 1.0) > RIGID_TRANSFORM_ATOL)
      throw PoseBatchInputError("target must be a finite rigid transform");
    owned_targets.push_back(target);
  }
  Eigen::MatrixXd output(rows.rows(), 2);
  {
    // All matrices/names are owned copies. The caller must keep topology fixed and
    // exclusively own the solver; no Python callback can execute during the loop.
    nb::gil_scoped_release release;
    for (Eigen::Index i = 0; i < rows.rows(); ++i)
    {
      ts::SceneState::JointValues held;
      for (const Eigen::Index index : held_indices)
        held.emplace(ids[static_cast<std::size_t>(index)], rows(i, index));
      const auto held_state = solver->getState(held);
      const Eigen::VectorXd q = rows.row(i).transpose();
      const auto full_state = solver->getState(ids, q);
      const Eigen::Matrix4d world_target = held_state.link_transforms.at(workpiece_id).matrix() *
                                           owned_targets[static_cast<std::size_t>(i)];
      const Eigen::Matrix4d world_tcp = full_state.link_transforms.at(tcp_id).matrix();
      output(i, 0) = (world_tcp.block<3, 1>(0, 3) - world_target.block<3, 1>(0, 3)).norm();
      const Eigen::Quaterniond a(world_tcp.block<3, 3>(0, 0));
      const Eigen::Quaterniond b(world_target.block<3, 3>(0, 0));
      output(i, 1) = a.angularDistance(b);
      if (!std::isfinite(output(i, 0)) || !std::isfinite(output(i, 1)))
        throw PoseBatchInputError("nonfinite residual");
    }
  }
  return output;
}
}  // namespace

void bind_workpiece_pose_rows(nb::module_& module)
{
  nb::exception<PoseBatchInputError>(module, "PoseBatchInputError");
  nb::exception<PoseBatchContextError>(module, "PoseBatchContextError");
  module.def("workpiece_pose_residual_rows", &residuals,
             nb::arg("solver"), nb::arg("joint_names"), nb::arg("held_joint_names"),
             nb::arg("rows"), nb::arg("targets"), nb::arg("workpiece_link"), nb::arg("tcp_link"),
             "Ordered metre/radian residuals; targets are row-major 4x4 matrices in the workpiece frame. "
             "Requires an exclusively owned OFKT solver with fixed topology during the call.");
}
