// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Evaluation wrappers for GPU math functions.

#ifndef GPU_LIBM_BENCH_KERNELS_COMMON_EVALUATE_HPP
#define GPU_LIBM_BENCH_KERNELS_COMMON_EVALUATE_HPP

#include "GPU.hpp"

namespace gpu_libm_bench {

template <auto Function, typename OutputType, typename... InputTypes>
void evaluate(uint64_t Count, OutputType *Output, const InputTypes *...Inputs) {
  uint64_t Index = gpu::globalThreadId();
  if (Index < Count)
    Output[Index] = Function(Inputs[Index]...);
}

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_EVALUATE_HPP
