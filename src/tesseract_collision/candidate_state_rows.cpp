#include "batch_state_errors.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tesseract/collision/bullet/bullet_discrete_bvh_manager.h>
#include <tesseract/collision/discrete_contact_manager.h>
#include <tesseract/common/contact_allowed_validator.h>
#include <tesseract/environment/environment.h>
#include <tesseract/state_solver/ofkt/ofkt_state_solver.h>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
namespace tc = tesseract::collision;
namespace ts = tesseract::scene_graph;

// Registered in batch_state_checks.cpp; exception types are shared across APIs.

void bind_candidate_state_rows(nb::module_& module)
{
  module.def("classify_candidate_state_rows",
    [](std::shared_ptr<ts::StateSolver> solver,
       std::shared_ptr<tc::DiscreteContactManager> manager,
       std::vector<std::string> names,
       Eigen::MatrixXd rows,
       Eigen::MatrixXd limits) -> std::vector<std::string> {
      if (!solver || !manager || names.empty() || rows.rows() == 0 ||
          rows.cols() != static_cast<long>(names.size()) || limits.rows() != static_cast<long>(names.size()) ||
          limits.cols() != 2 || !rows.allFinite() || !limits.allFinite() ||
          (limits.col(0).array() > limits.col(1).array()).any())
        throw InvalidBatch("invalid batch rows, limits, solver, or manager");
      const auto& solver_ref = *solver.get();
      const auto& manager_ref = *manager.get();
      if (typeid(solver_ref) != typeid(ts::OFKTStateSolver) || typeid(manager_ref) != typeid(tc::BulletDiscreteBVHManager))
        throw UnsupportedContext("only native OFKT and Bullet BVH are supported");
      const auto validator = manager->getContactAllowedValidator();
      if (validator)
      {
        const auto& validator_ref = *validator.get();
        if (typeid(validator_ref) != typeid(tesseract::environment::EnvironmentContactAllowedValidator) &&
            typeid(validator_ref) != typeid(tesseract::common::ACMContactAllowedValidator))
          throw UnsupportedContext("unsupported contact validator or callback context");
      }
      std::set<std::string> unique;
      std::vector<tesseract::common::JointId> ids;
      const auto available = solver->getJointIds();
      for (const auto& name : names)
      {
        if (name.empty() || !unique.insert(name).second)
          throw InvalidBatch("joint names must be nonempty and unique");
        tesseract::common::JointId id(name);
        if (std::find(available.begin(), available.end(), id) == available.end())
          throw InvalidBatch("joint is not present in the state solver");
        ids.push_back(std::move(id));
      }
      std::vector<std::string> verdicts;
      verdicts.reserve(static_cast<std::size_t>(rows.rows()));
      {
        nb::gil_scoped_release release;
        for (Eigen::Index i = 0; i < rows.rows(); ++i)
        {
          Eigen::VectorXd q = rows.row(i).transpose();
          if ((q.array() < limits.col(0).array()).any() || (q.array() > limits.col(1).array()).any())
          {
            verdicts.emplace_back("joint_limit");
            continue;
          }
          const auto state = solver->getState(ids, q);
          manager->setCollisionObjectsTransform(state.link_transforms);
          tc::ContactResultMap contacts;
          manager->contactTest(contacts, tc::ContactRequest(tc::ContactTestType::ALL));
          tc::ContactResultVector flat;
          contacts.flattenMoveResults(flat);
          const bool penetrating = std::any_of(flat.begin(), flat.end(), [](const auto& hit) { return hit.distance < 0; });
          verdicts.emplace_back(penetrating ? "sampled_collision" : "clear");
        }
      }
      return verdicts;
    }, nb::arg("solver"), nb::arg("manager"), nb::arg("names"), nb::arg("rows"), nb::arg("limits"),
    "Classify ordered full-cell candidate rows on an exclusively owned native solver and manager. "
    "The caller applies ACM and margins and checks its budget before and after each bounded call. "
    "Results are clear, joint_limit or sampled_collision; no collision details are returned.");
}
