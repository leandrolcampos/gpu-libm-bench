// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Device kernel launch configurations.

#ifndef GPU_LIBM_BENCH_COMMON_LAUNCH_HPP
#define GPU_LIBM_BENCH_COMMON_LAUNCH_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace gpu_libm_bench {

struct Dim3 {
  std::uint32_t X = 1;
  std::uint32_t Y = 1;
  std::uint32_t Z = 1;
};

struct LaunchConfig {
  Dim3 GridDim;
  Dim3 BlockDim;
  std::size_t SharedMemoryBytes = 0;

  bool isValid() const {
    return GridDim.X && GridDim.Y && GridDim.Z && BlockDim.X && BlockDim.Y &&
           BlockDim.Z &&
           SharedMemoryBytes <= std::numeric_limits<unsigned int>::max();
  }
};

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_COMMON_LAUNCH_HPP
