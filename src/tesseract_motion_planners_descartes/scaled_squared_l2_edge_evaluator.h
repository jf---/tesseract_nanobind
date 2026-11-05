/**
 * @file scaled_squared_l2_edge_evaluator.h
 * @brief Descartes edge cost sum_j ((end_j - start_j) / scale_j)^2, always valid.
 *
 * With scale = the joint velocity limits this is the time-normalised squared L2: every axis is
 * priced in seconds, so no axis moves for free. descartes_light's EuclideanDistanceEdgeEvaluator
 * multiplies by a scale and ignores it when its size differs from the state's; this one divides
 * (the metric's own operation), raises on a size mismatch, and sums sequentially in joint order
 * with floating-point contraction off, so for dof <= 7 it equals numpy's
 * `(((end - start) / scale) ** 2).sum()` bit for bit (numpy sums arrays shorter than 8 in order).
 */
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include <descartes_light/core/edge_evaluator.h>
#include <descartes_light/types.h>

namespace tesseract_nb {

/** The evaluated states' dof differs from the evaluator's scale. */
class EdgeDofMismatchError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

/** A scale entry is not a positive finite number. */
class EdgeScaleError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

class EdgeCostError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

template <typename FloatType>
class ScaledSquaredL2EdgeEvaluator : public descartes_light::EdgeEvaluator<FloatType> {
public:
    using VectorX = Eigen::Matrix<FloatType, Eigen::Dynamic, 1>;

    explicit ScaledSquaredL2EdgeEvaluator(const VectorX& scale) : scale_(scale) {
        if (scale_.size() == 0)
            throw EdgeScaleError("scale must not be empty");
        for (Eigen::Index j = 0; j < scale_.size(); ++j)
            if (!(std::isfinite(scale_[j]) && scale_[j] > 0))
                throw EdgeScaleError("scale[" + std::to_string(j) + "] must be positive and finite");
    }

    std::pair<bool, FloatType> evaluate(const descartes_light::State<FloatType>& start,
                                        const descartes_light::State<FloatType>& end) const override {
#pragma clang fp contract(off)
        const Eigen::Index n = scale_.size();
        if (start.values.size() != n || end.values.size() != n)
            throw EdgeDofMismatchError("state dof " + std::to_string(start.values.size()) + " / " +
                                       std::to_string(end.values.size()) + " differs from scale size " +
                                       std::to_string(n));
        FloatType cost = 0;
        for (Eigen::Index j = 0; j < n; ++j) {
            if (!std::isfinite(start.values[j]) || !std::isfinite(end.values[j]))
                throw EdgeCostError("state values must be finite");
            const FloatType t = (end.values[j] - start.values[j]) / scale_[j];
            const FloatType sq = t * t;
            cost = cost + sq;
        }
        if (!std::isfinite(cost))
            throw EdgeCostError("scaled squared edge cost must be finite");
        return {true, cost};
    }

    const VectorX& scale() const { return scale_; }

private:
    VectorX scale_;
};

}  // namespace tesseract_nb
