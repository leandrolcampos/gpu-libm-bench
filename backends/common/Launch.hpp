//===----------------------------------------------------------------------===//
// Copyright 2026 Leandro Lacerda Campos
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Device kernel launch configurations.
//
//===----------------------------------------------------------------------===//

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
