/**
 * @file bgl_dijkstra_solver_profile.h
 * @brief A Descartes solver profile for descartes_light's BGL Dijkstra solver with dynamic edges.
 *
 * `BGLDijkstraSVDESolver` builds every vertex (static vertices) but adds a vertex's out-edges only
 * when Dijkstra examines it (dynamic edges), and stops at the first last-rung vertex it examines.
 * With non-negative edge and vertex costs that vertex is optimal, so the path's cost equals the
 * ladder-graph DP's; the edges out of vertices never examined (cost-to-come above the optimum, or
 * unreachable) are never evaluated. Equal-cost ties may resolve to a different path than the
 * ladder solver's.
 *
 * The early terminator needs the last rung's index, which is known only at build time, so the
 * profile's solver is a thin wrapper that constructs the BGL solver in `buildImpl`. The wrapper
 * also wraps each edge evaluator in a counter, so the profile reports how many edges the last
 * solve evaluated, alongside the vertices and the edges the graph kept.
 *
 * Not the depth-first variant: it stops at the first last-rung vertex it discovers, which is a
 * path, not the cheapest one.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <descartes_light/bgl/bgl_dijkstra_solver.h>
#include <descartes_light/bgl/event_visitors.h>
#include <descartes_light/core/edge_evaluator.h>
#include <descartes_light/core/solver.h>
#include <descartes_light/core/state_evaluator.h>
#include <descartes_light/core/waypoint_sampler.h>
#include <tesseract/motion_planners/descartes/profile/descartes_profile.h>

namespace tesseract_nb {

/** What the last solve built and evaluated (shared by a profile and the solvers it creates). */
struct BGLSolveStats {
    std::atomic<std::uint64_t> edges_evaluated{0};
    std::atomic<std::uint64_t> graph_edges{0};
    std::atomic<std::uint64_t> graph_vertices{0};
    std::atomic<std::int64_t> build_ns{0};
    std::atomic<std::int64_t> search_ns{0};
    // 0 is valid; 1 sampler, 2 state, 3 edge, 4 vertex, 5 combined edge, 6 path.
    std::atomic<int> invalid_cost{0};
};

class InvalidBGLCost : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

class BGLThreadCountError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

template <typename FloatType>
bool invalidCost(FloatType cost) { return !std::isfinite(cost) || cost < static_cast<FloatType>(0); }

template <typename FloatType>
class CheckedSampler : public descartes_light::WaypointSampler<FloatType> {
public:
    CheckedSampler(typename descartes_light::WaypointSampler<FloatType>::ConstPtr inner,
                   std::shared_ptr<BGLSolveStats> stats) : inner_(std::move(inner)), stats_(std::move(stats)) {}

    std::vector<descartes_light::StateSample<FloatType>> sample() const override {
        auto samples = inner_->sample();
        for (const auto& sample : samples) {
            if (invalidCost(sample.cost)) {
                stats_->invalid_cost.store(1, std::memory_order_relaxed);
                return {};
            }
        }
        return samples;
    }

private:
    typename descartes_light::WaypointSampler<FloatType>::ConstPtr inner_;
    std::shared_ptr<BGLSolveStats> stats_;
};

template <typename FloatType>
class CheckedStateEvaluator : public descartes_light::StateEvaluator<FloatType> {
public:
    CheckedStateEvaluator(typename descartes_light::StateEvaluator<FloatType>::ConstPtr inner,
                          std::shared_ptr<BGLSolveStats> stats) : inner_(std::move(inner)), stats_(std::move(stats)) {}

    std::pair<bool, FloatType> evaluate(const descartes_light::State<FloatType>& state) const override {
        auto result = inner_->evaluate(state);
        if (result.first && invalidCost(result.second)) {
            stats_->invalid_cost.store(2, std::memory_order_relaxed);
            return {false, static_cast<FloatType>(0)};
        }
        return result;
    }

private:
    typename descartes_light::StateEvaluator<FloatType>::ConstPtr inner_;
    std::shared_ptr<BGLSolveStats> stats_;
};

/** Counts every evaluate call of the evaluator it wraps. */
template <typename FloatType>
class CountingEdgeEvaluator : public descartes_light::EdgeEvaluator<FloatType> {
public:
    CountingEdgeEvaluator(typename descartes_light::EdgeEvaluator<FloatType>::ConstPtr inner,
                          std::shared_ptr<BGLSolveStats> stats)
      : inner_(std::move(inner)), stats_(std::move(stats)) {}

    std::pair<bool, FloatType> evaluate(const descartes_light::State<FloatType>& start,
                                        const descartes_light::State<FloatType>& end) const override {
        stats_->edges_evaluated.fetch_add(1, std::memory_order_relaxed);
        auto result = inner_->evaluate(start, end);
        if (result.first && invalidCost(result.second)) {
            stats_->invalid_cost.store(3, std::memory_order_relaxed);
            return {false, static_cast<FloatType>(0)};
        }
        return result;
    }

private:
    typename descartes_light::EdgeEvaluator<FloatType>::ConstPtr inner_;
    std::shared_ptr<BGLSolveStats> stats_;
};

template <typename FloatType>
class BGLDijkstraSVDEWrapper : public descartes_light::Solver<FloatType> {
public:
    using Terminator = descartes_light::early_terminator<boost::on_examine_vertex>;
    using Impl = descartes_light::BGLDijkstraSVDESolver<FloatType, Terminator>;

    BGLDijkstraSVDEWrapper(unsigned num_threads, std::shared_ptr<BGLSolveStats> stats)
      : num_threads_(num_threads), stats_(std::move(stats)) {}

    descartes_light::SearchResult<FloatType> search() override {
        if (!impl_)
            throw std::logic_error("BGLDijkstraSVDEWrapper: search() before build()");
        const auto started = std::chrono::steady_clock::now();
        descartes_light::SearchResult<FloatType> result;
        try {
            result = impl_->search();
        } catch (...) {
            checkCombinedEdgeCosts();
            if (stats_->invalid_cost.load(std::memory_order_relaxed) != 0)
                throw InvalidBGLCost("BGL costs must be finite and nonnegative");
            throw;
        }
        checkCombinedEdgeCosts();
        if (invalidCost(result.cost))
            stats_->invalid_cost.store(6, std::memory_order_relaxed);
        if (stats_->invalid_cost.load(std::memory_order_relaxed) != 0)
            throw InvalidBGLCost("BGL costs must be finite and nonnegative");
        stats_->search_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count(), std::memory_order_relaxed);
        stats_->graph_edges.store(boost::num_edges(impl_->getGraph()), std::memory_order_relaxed);
        return result;
    }

protected:
    descartes_light::BuildStatus
    buildImpl(const std::vector<typename descartes_light::WaypointSampler<FloatType>::ConstPtr>& trajectory,
              const std::vector<typename descartes_light::EdgeEvaluator<FloatType>::ConstPtr>& edge_evaluators,
              const std::vector<typename descartes_light::StateEvaluator<FloatType>::ConstPtr>& state_evaluators) override {
        if (trajectory.empty())
            throw std::invalid_argument("BGLDijkstraSVDEWrapper: empty trajectory");
        stats_->edges_evaluated.store(0, std::memory_order_relaxed);
        stats_->graph_edges.store(0, std::memory_order_relaxed);
        stats_->invalid_cost.store(0, std::memory_order_relaxed);
        std::vector<typename descartes_light::WaypointSampler<FloatType>::ConstPtr> checked_trajectory;
        checked_trajectory.reserve(trajectory.size());
        for (const auto& sampler : trajectory)
            checked_trajectory.push_back(std::make_shared<CheckedSampler<FloatType>>(sampler, stats_));
        std::vector<typename descartes_light::StateEvaluator<FloatType>::ConstPtr> checked_states;
        checked_states.reserve(state_evaluators.size());
        for (const auto& state_evaluator : state_evaluators)
            checked_states.push_back(std::make_shared<CheckedStateEvaluator<FloatType>>(state_evaluator, stats_));
        std::vector<typename descartes_light::EdgeEvaluator<FloatType>::ConstPtr> counted;
        counted.reserve(edge_evaluators.size());
        for (const auto& e : edge_evaluators)
            counted.push_back(std::make_shared<CountingEdgeEvaluator<FloatType>>(e, stats_));
        impl_ = std::make_unique<Impl>(Terminator(static_cast<long>(trajectory.size()) - 1), num_threads_);
        const auto started = std::chrono::steady_clock::now();
        auto status = impl_->build(checked_trajectory, counted, checked_states);
        const auto vertices = boost::vertices(impl_->getGraph());
        for (auto it = vertices.first; it != vertices.second; ++it) {
            if (invalidCost(impl_->getGraph()[*it].sample.cost))
                stats_->invalid_cost.store(4, std::memory_order_relaxed);
        }
        if (stats_->invalid_cost.load(std::memory_order_relaxed) != 0)
            throw InvalidBGLCost("BGL costs must be finite and nonnegative");
        stats_->build_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count(), std::memory_order_relaxed);
        stats_->graph_vertices.store(boost::num_vertices(impl_->getGraph()), std::memory_order_relaxed);
        return status;
    }

private:
    void checkCombinedEdgeCosts() const {
        const auto& graph = impl_->getGraph();
        const auto edges = boost::edges(graph);
        const auto weights = boost::get(boost::edge_weight, graph);
        for (auto it = edges.first; it != edges.second; ++it) {
            if (invalidCost(boost::get(weights, *it)))
                stats_->invalid_cost.store(5, std::memory_order_relaxed);
        }
    }

    unsigned num_threads_;
    std::shared_ptr<BGLSolveStats> stats_;
    std::unique_ptr<Impl> impl_;
};

template <typename FloatType>
class DescartesBGLDijkstraSolverProfile : public tesseract::motion_planners::DescartesSolverProfile<FloatType> {
public:
    /**
     * Threads of the vertex build. BGLSolverBaseSVDE::buildImpl (descartes_light 0.4.10) calls
     * boost::add_vertex on the shared graph inside its OpenMP loop without a lock, so more than one
     * thread is a data race: the profile refuses it.
     */
    int num_threads{1};

    std::unique_ptr<descartes_light::Solver<FloatType>> create() const override {
        if (num_threads != 1)
            throw BGLThreadCountError(
                "DescartesBGLDijkstraSolverProfile: num_threads must be 1 (descartes_light's SVDE vertex "
                "build adds vertices to a shared graph without a lock)");
        return std::make_unique<BGLDijkstraSVDEWrapper<FloatType>>(1U, stats_);
    }

    std::uint64_t edgesEvaluated() const { return stats_->edges_evaluated.load(std::memory_order_relaxed); }
    std::uint64_t graphEdges() const { return stats_->graph_edges.load(std::memory_order_relaxed); }
    std::uint64_t graphVertices() const { return stats_->graph_vertices.load(std::memory_order_relaxed); }
    std::int64_t buildNanoseconds() const { return stats_->build_ns.load(std::memory_order_relaxed); }
    std::int64_t searchNanoseconds() const { return stats_->search_ns.load(std::memory_order_relaxed); }
    int invalidCostKind() const { return stats_->invalid_cost.load(std::memory_order_relaxed); }

private:
    std::shared_ptr<BGLSolveStats> stats_ = std::make_shared<BGLSolveStats>();
};

}  // namespace tesseract_nb
