// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Kernel profiling metrics and results.

#ifndef GPU_LIBM_BENCH_COMMON_PROFILING_HPP
#define GPU_LIBM_BENCH_COMMON_PROFILING_HPP

#include <cstdint>
#include <optional>

namespace gpu_libm_bench {

enum class KernelMetric : std::uint8_t {
  ClockFrequency,
  Duration,
};

struct KernelProfile {
  std::optional<double> ClockFrequencyHz;
  std::optional<double> DurationMs;
};

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_COMMON_PROFILING_HPP
