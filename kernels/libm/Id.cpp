// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Kernel entry points for the double identity.

extern "C" inline double gpu_libm_bench_id(double X) { return X; }

#define LIBM_NAME id
#define LIBM_SYMBOL gpu_libm_bench_id
#define LIBM_ARG_TYPE double
#define LIBM_RETURN_TYPE double

#include "UnaryKernels.inc" // IWYU pragma: keep
