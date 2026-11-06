#pragma once

#include <chrono>
#include <vector>

#include <trajopt_sqp/piqp_solver.h>

// Research-only wrapper: the underlying PIQP solver retains its existing behavior.
class PIQPTimedSolver final : public trajopt_sqp::PIQPSolver {
public:
    struct Record {
        bool success = false;
        bool native_timing_available = false;
        bool reused_setup = false;
        std::chrono::duration<double> wall{0};
        std::chrono::duration<double> setup{0};
        std::chrono::duration<double> update{0};
        std::chrono::duration<double> solve{0};
    };

    PIQPTimedSolver() { settings.compute_timings = true; }

    bool solve() override {
        const auto started = std::chrono::steady_clock::now();
        Record record;
        try {
            record.success = trajopt_sqp::PIQPSolver::solve();
        } catch (...) {
            record.wall = std::chrono::steady_clock::now() - started;
            records_.push_back(record);
            throw;
        }
        record.wall = std::chrono::steady_clock::now() - started;
        // A rejected call may not reach PIQP, so its result could be from a prior call.
        if (record.success && settings.compute_timings) {
            const auto& info = solver().result().info;
            record.native_timing_available = true;
            record.reused_setup = reusedSetup();
            record.setup = std::chrono::duration<double>(info.setup_time);
            record.update = std::chrono::duration<double>(info.update_time);
            record.solve = std::chrono::duration<double>(info.solve_time);
        }
        records_.push_back(record);
        return record.success;
    }

    const std::vector<Record>& timingRecords() const { return records_; }
    void clearTimingRecords() { records_.clear(); }

private:
    std::vector<Record> records_;
};
