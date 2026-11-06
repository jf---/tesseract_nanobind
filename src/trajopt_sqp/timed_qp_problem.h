#pragma once

#include <chrono>
#include <cstddef>

#include <trajopt_sqp/trajopt_qp_problem.h>

// Measures existing QPProblem entry points without changing their computation.
class TimedTrajOptQPProblem final : public trajopt_sqp::TrajOptQPProblem {
public:
    using trajopt_sqp::TrajOptQPProblem::TrajOptQPProblem;
    TimedTrajOptQPProblem(const TimedTrajOptQPProblem&) = delete;
    TimedTrajOptQPProblem& operator=(const TimedTrajOptQPProblem&) = delete;
    TimedTrajOptQPProblem(TimedTrajOptQPProblem&&) = delete;
    TimedTrajOptQPProblem& operator=(TimedTrajOptQPProblem&&) = delete;
    ~TimedTrajOptQPProblem() override = default;

    struct Phase {
        std::size_t calls = 0;
        std::chrono::duration<double> elapsed{0};
    };

    void setVariables(const double* values) override {
        Timer timer(set_variables_);
        trajopt_sqp::TrajOptQPProblem::setVariables(values);
    }
    void convexify() override {
        Timer timer(convexify_);
        trajopt_sqp::TrajOptQPProblem::convexify();
    }
    Eigen::VectorXd getExactCosts() const override {
        Timer timer(exact_costs_);
        return trajopt_sqp::TrajOptQPProblem::getExactCosts();
    }
    trajopt_sqp::ConstraintViolations getExactConstraintViolations() const override {
        Timer timer(exact_constraints_);
        return trajopt_sqp::TrajOptQPProblem::getExactConstraintViolations();
    }
    Eigen::VectorXd evaluateConvexCosts(const Eigen::Ref<const Eigen::VectorXd>& values) const override {
        Timer timer(convex_costs_);
        return trajopt_sqp::TrajOptQPProblem::evaluateConvexCosts(values);
    }
    trajopt_sqp::ConstraintViolations
    evaluateConvexConstraintViolations(const Eigen::Ref<const Eigen::VectorXd>& values) const override {
        Timer timer(convex_constraints_);
        return trajopt_sqp::TrajOptQPProblem::evaluateConvexConstraintViolations(values);
    }

    const Phase& setVariablesTiming() const { return set_variables_; }
    const Phase& convexifyTiming() const { return convexify_; }
    const Phase& exactCostsTiming() const { return exact_costs_; }
    const Phase& exactConstraintsTiming() const { return exact_constraints_; }
    const Phase& convexCostsTiming() const { return convex_costs_; }
    const Phase& convexConstraintsTiming() const { return convex_constraints_; }

private:
    struct Timer {
        explicit Timer(Phase& current) : phase(current), started(std::chrono::steady_clock::now()) {
            ++phase.calls;
        }
        ~Timer() { phase.elapsed += std::chrono::steady_clock::now() - started; }
        Phase& phase;
        std::chrono::steady_clock::time_point started;
    };

    Phase set_variables_;
    Phase convexify_;
    mutable Phase exact_costs_;
    mutable Phase exact_constraints_;
    mutable Phase convex_costs_;
    mutable Phase convex_constraints_;
};
