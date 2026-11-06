#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <tesseract/kinematics/inverse_kinematics.h>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace nb = nanobind;
namespace tc = tesseract::common;
namespace tk = tesseract::kinematics;

struct Gate {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered{false};
  bool released{false};
  bool fail{false};
};

class BlockingInverse : public tk::InverseKinematics {
public:
  BlockingInverse() : gate_(std::make_shared<Gate>()), joints_{tc::JointId("joint")} {}
  explicit BlockingInverse(const std::vector<std::string>& names) : gate_(std::make_shared<Gate>()) {
    for (const auto& name : names) joints_.emplace_back(name);
  }
  BlockingInverse(std::shared_ptr<Gate> gate, std::vector<tc::JointId> joints)
      : gate_(std::move(gate)), joints_(std::move(joints)) {}
  void calcInvKin(tk::IKSolutions& out, const tc::LinkIdTransformMap&,
                  const Eigen::Ref<const Eigen::VectorXd>& seed) const override {
    std::unique_lock<std::mutex> lock(gate_->mutex);
    gate_->entered = true;
    gate_->cv.notify_all();
    if (!gate_->cv.wait_for(lock, std::chrono::seconds(5), [&] { return gate_->released; }))
      throw std::runtime_error("probe gate timed out");
    if (gate_->fail) throw std::runtime_error("probe solver failure");
    out.emplace_back(seed);
  }
  bool awaitEntry() {
    std::unique_lock<std::mutex> lock(gate_->mutex);
    return gate_->cv.wait_for(lock, std::chrono::seconds(5), [&] { return gate_->entered; });
  }
  void unblock() {
    std::lock_guard<std::mutex> lock(gate_->mutex);
    gate_->released = true;
    gate_->cv.notify_all();
  }
  void fail() {
    std::lock_guard<std::mutex> lock(gate_->mutex);
    gate_->fail = true;
    gate_->released = true;
    gate_->cv.notify_all();
  }
  std::vector<tc::JointId> getJointIds() const override { return joints_; }
  Eigen::Index numJoints() const override { return static_cast<Eigen::Index>(joints_.size()); }
  tc::LinkId getBaseLinkId() const override { return tc::LinkId("base_link"); }
  tc::LinkId getWorkingFrame() const override { return tc::LinkId("base_link"); }
  std::vector<tc::LinkId> getTipLinkIds() const override { return {tc::LinkId("tool0")}; }
  std::string getSolverName() const override { return "blocking test solver"; }
  tk::InverseKinematics::UPtr clone() const override { return std::make_unique<BlockingInverse>(gate_, joints_); }
private:
  std::shared_ptr<Gate> gate_;
  std::vector<tc::JointId> joints_;
};

NB_MODULE(ik_ownership_probe, m) {
  nb::module_::import_("tesseract_robotics.tesseract_kinematics");
  nb::class_<BlockingInverse, tk::InverseKinematics>(m, "BlockingInverse")
    .def(nb::init<>())
    .def(nb::init<const std::vector<std::string>&>())
    .def("await_entry", &BlockingInverse::awaitEntry, nb::call_guard<nb::gil_scoped_release>())
    .def("unblock", &BlockingInverse::unblock)
    .def("fail", &BlockingInverse::fail);
}
