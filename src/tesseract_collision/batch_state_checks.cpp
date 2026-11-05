#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <tesseract/collision/discrete_contact_manager.h>
#include <tesseract/collision/bullet/bullet_discrete_bvh_manager.h>
#include <tesseract/common/contact_allowed_validator.h>
#include <tesseract/environment/environment.h>
#include <tesseract/state_solver/ofkt/ofkt_state_solver.h>
#include <tesseract/collision/types.h>
#include <tesseract/state_solver/state_solver.h>
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace nb = nanobind;
namespace tc = tesseract::collision;
namespace ts = tesseract::scene_graph;
struct InvalidBatch : std::runtime_error { using std::runtime_error::runtime_error; };
struct UnsupportedContext : std::runtime_error { using std::runtime_error::runtime_error; };
using Result = std::tuple<std::string, long, std::string, std::string, double>;

void bind_batch_state_checks(nb::module_& module)
{
  nb::module_::import_("tesseract_robotics.tesseract_state_solver");
  nb::exception<InvalidBatch>(module, "InvalidBatch");
  nb::exception<UnsupportedContext>(module, "UnsupportedBatchContext");
  module.def("check_batched_state_rows", [](std::shared_ptr<ts::StateSolver> solver,
                              std::shared_ptr<tc::DiscreteContactManager> manager,
                              const std::vector<std::string>& names,
                              Eigen::MatrixXd rows,
                              Eigen::MatrixXd limits) -> Result {
    if (!solver || !manager || names.empty() || rows.rows() == 0 || rows.cols() != static_cast<long>(names.size()) ||
        limits.rows() != static_cast<long>(names.size()) || limits.cols() != 2 || !rows.allFinite() ||
        !limits.allFinite() || (limits.col(0).array() > limits.col(1).array()).any())
      throw InvalidBatch("invalid batch rows, limits, solver, or manager");
    const auto& solver_ref = *solver.get();
    const auto& manager_ref = *manager.get();
    if (typeid(solver_ref) != typeid(ts::OFKTStateSolver) ||
        typeid(manager_ref) != typeid(tc::BulletDiscreteBVHManager))
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
    Result result("clear", -1, "", "", 0.0);
    {
      nb::gil_scoped_release release;
      for (Eigen::Index i = 0; i < rows.rows(); ++i)
      {
        Eigen::VectorXd q = rows.row(i).transpose();
        if ((q.array() < limits.col(0).array()).any() || (q.array() > limits.col(1).array()).any())
        {
          result = Result("joint_limit", static_cast<long>(i), "", "", 0.0);
          break;
        }
        const auto state = solver->getState(ids, q);
        manager->setCollisionObjectsTransform(state.link_transforms);
        tc::ContactResultMap contacts;
        manager->contactTest(contacts, tc::ContactRequest(tc::ContactTestType::ALL));
        tc::ContactResultVector flat;
        contacts.flattenMoveResults(flat);
        const tc::ContactResult* worst = nullptr;
        for (const auto& hit : flat)
          if (hit.distance < 0 && (worst == nullptr || hit.distance < worst->distance))
            worst = &hit;
        if (worst != nullptr)
        {
          auto a = worst->link_ids[0].name();
          auto b = worst->link_ids[1].name();
          if (a > b) std::swap(a, b);
          result = Result("sampled_collision", static_cast<long>(i), a, b, worst->distance);
          break;
        }
      }
    }
    return result;
  }, nb::arg("solver"), nb::arg("manager"), nb::arg("names"), nb::arg("rows"), nb::arg("limits"),
     "Check ordered joint rows using a worker-owned solver and manager. "
     "The caller must apply its ACM and margin first and must not share or mutate the manager during the call. "
     "Returns the first joint-limit or negative-contact failure; unsupported native contexts fail loudly.");
}
