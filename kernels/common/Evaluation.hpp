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
// Evaluation wrappers for GPU math functions.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_KERNELS_COMMON_EVALUATION_HPP
#define GPU_LIBM_BENCH_KERNELS_COMMON_EVALUATION_HPP

#include "GPU.hpp"

#include <stdint.h>

namespace gpu_libm_bench {

template <auto Function, typename OutputType, typename... InputTypes>
void evaluate(uint64_t Count, OutputType *Output, const InputTypes *...Inputs) {
  uint64_t Index = gpu::globalThreadId();
  if (Index < Count)
    Output[Index] = Function(Inputs[Index]...);
}

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_EVALUATION_HPP
