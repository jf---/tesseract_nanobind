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

static std::vector<std::string> classify_rows(std::shared_ptr<ts::StateSolver> solver,
                                              std::shared_ptr<tc::DiscreteContactManager> manager,
                                              std::vector<std::string> names,
                                              Eigen::MatrixXd rows,
                                              Eigen::MatrixXd limits,
                                              bool first_screen,
                                              bool check_limits)
{
      if (!solver || !manager || names.empty() || rows.rows() == 0 ||
          rows.cols() != static_cast<long>(names.size()) || !rows.allFinite() ||
          (check_limits && (limits.rows() != static_cast<long>(names.size()) || limits.cols() != 2 ||
                            !limits.allFinite() || (limits.col(0).array() > limits.col(1).array()).any())))
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
          if (check_limits && ((q.array() < limits.col(0).array()).any() || (q.array() > limits.col(1).array()).any()))
          {
            verdicts.emplace_back("joint_limit");
            continue;
          }
          const auto state = solver->getState(ids, q);
          manager->setCollisionObjectsTransform(state.link_transforms);
          tc::ContactResultMap contacts;
          tc::ContactResultVector flat;
          bool penetrating = false;
          bool require_all = true;
          if (first_screen)
          {
            manager->contactTest(contacts, tc::ContactRequest(tc::ContactTestType::FIRST));
            contacts.flattenMoveResults(flat);
            if (flat.empty())
              require_all = false;
            else if (flat.front().distance < 0)
            {
              penetrating = true;
              require_all = false;
            }
            contacts.clear();
            flat.clear();
          }
          if (require_all)
          {
            manager->contactTest(contacts, tc::ContactRequest(tc::ContactTestType::ALL));
            contacts.flattenMoveResults(flat);
            penetrating = std::any_of(flat.begin(), flat.end(), [](const auto& hit) { return hit.distance < 0; });
          }
          verdicts.emplace_back(penetrating ? "sampled_collision" : "clear");
        }
      }
      return verdicts;
 }

void bind_candidate_state_rows(nb::module_& module)
{
  auto bind = [&module](const char* name, bool first_screen) {
    module.def(name,
      [first_screen](std::shared_ptr<ts::StateSolver> solver,
                     std::shared_ptr<tc::DiscreteContactManager> manager,
                     std::vector<std::string> names,
                     Eigen::MatrixXd rows,
                     Eigen::MatrixXd limits) {
        return classify_rows(std::move(solver), std::move(manager), std::move(names),
                             std::move(rows), std::move(limits), first_screen, true);
      }, nb::arg("solver"), nb::arg("manager"), nb::arg("names"), nb::arg("rows"), nb::arg("limits"),
      "Classify ordered full-cell candidates using an exclusively owned solver and manager. "
      "Apply ACM and margins before each call; check deadline before and after bounded calls.");
  };
  bind("classify_candidate_state_rows", false);
  bind("classify_candidate_state_rows_first", true);
  module.def("classify_collision_state_rows_first",
    [](std::shared_ptr<ts::StateSolver> solver,
       std::shared_ptr<tc::DiscreteContactManager> manager,
       std::vector<std::string> names,
       Eigen::MatrixXd rows) {
      return classify_rows(std::move(solver), std::move(manager), std::move(names),
                           std::move(rows), Eigen::MatrixXd(), true, false);
    }, nb::arg("solver"), nb::arg("manager"), nb::arg("names"), nb::arg("rows"),
    "Classify collision for every finite ordered row without a joint-limit screen. "
    "The caller owns its solver and manager and applies ACM and margins before each call.");
}
