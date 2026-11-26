// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Kernel entry points for the float identity.

extern "C" inline float gpu_libm_bench_idf(float X) { return X; }

#define LIBM_NAME idf
#define LIBM_SYMBOL gpu_libm_bench_idf
#define LIBM_ARG_TYPE float
#define LIBM_RETURN_TYPE float

#include "UnaryKernels.inc" // IWYU pragma: keep
