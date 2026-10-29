// Copyright 2026 Leandro A. Lacerda Campos
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
  Dim3 Grid;
  Dim3 Block;
  std::size_t SharedMemoryBytes = 0;

  bool isValid() const {
    return Grid.X && Grid.Y && Grid.Z && Block.X && Block.Y && Block.Z &&
           SharedMemoryBytes <= std::numeric_limits<unsigned int>::max();
  }
};

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_COMMON_LAUNCH_HPP
