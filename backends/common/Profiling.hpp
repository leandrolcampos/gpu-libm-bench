// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0

#ifndef GPU_LIBM_BENCH_COMMON_PROFILING_HPP
#define GPU_LIBM_BENCH_COMMON_PROFILING_HPP

#include <cstdint>
#include <optional>

namespace gpu_libm_bench {

enum class KernelMetric : std::uint8_t {
  AverageFrequency,
  Duration,
};

struct KernelProfile {
  std::optional<double> AverageFrequencyHz;
  std::optional<double> DurationMs;
};

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_COMMON_PROFILING_HPP
