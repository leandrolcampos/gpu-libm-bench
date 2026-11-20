// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Helper macros for math kernels.

#ifndef GPU_LIBM_BENCH_KERNELS_LIBM_MACROS_HPP
#define GPU_LIBM_BENCH_KERNELS_LIBM_MACROS_HPP

#define LIBM_KERNEL_NAME_IMPL(Name, Suffix) Name##Suffix
#define LIBM_KERNEL_NAME(Name, Suffix) LIBM_KERNEL_NAME_IMPL(Name, Suffix)

#endif // GPU_LIBM_BENCH_KERNELS_LIBM_MACROS_HPP
