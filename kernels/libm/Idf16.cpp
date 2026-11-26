// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Kernel entry points for the _Float16 identity.

extern "C" inline _Float16 gpu_libm_bench_idf16(_Float16 X) { return X; }

#define LIBM_NAME idf16
#define LIBM_SYMBOL gpu_libm_bench_idf16
#define LIBM_ARG_TYPE _Float16
#define LIBM_RETURN_TYPE _Float16

#include "UnaryKernels.inc" // IWYU pragma: keep
