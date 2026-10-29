// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Kernel wrappers for LLVM-libm expf.

#include "Wrappers.hpp"

extern "C" float expf(float);

extern "C" __gpu_kernel void expfEvaluate(const float *X, float *Output,
                                          uint64_t Count) {
  gpu_libm_bench::evaluate<expf>(Count, Output, X);
}
