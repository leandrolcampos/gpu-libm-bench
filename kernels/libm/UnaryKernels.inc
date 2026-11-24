// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Kernel entry points for unary math functions.

#ifndef GPU_LIBM_BENCH_KERNELS_LIBM_UNARY_KERNELS_INC
#define GPU_LIBM_BENCH_KERNELS_LIBM_UNARY_KERNELS_INC

#ifdef LIBM_NAME
#include "Evaluation.hpp"
#include "Latency.hpp"
#include "Macros.hpp"
#include "Throughput.hpp"

namespace gpu_libm_bench::libm::LIBM_NAME {

using ArgType = LIBM_ARG_TYPE;
using ReturnType = LIBM_RETURN_TYPE;
using StorageType = gpu_libm_bench::fputil::FPBits<ArgType>::StorageType;

extern "C" ReturnType LIBM_SYMBOL(ArgType);

extern "C" __gpu_kernel void
LIBM_KERNEL_NAME(LIBM_NAME, Evaluation)(const ArgType *X, ReturnType *Output,
                                        uint64_t Count) {
  gpu_libm_bench::evaluate<LIBM_SYMBOL>(Count, Output, X);
}

extern "C" __gpu_kernel void LIBM_KERNEL_NAME(LIBM_NAME, Latency)(
    uint32_t PassCount, uint32_t ChainLength, const StorageType *ZeroMask,
    uint64_t *ElapsedTicks, StorageType *Output, const ArgType *X) {
  gpu_libm_bench::measureLatency<LIBM_SYMBOL>(PassCount, ChainLength, ZeroMask,
                                              ElapsedTicks, Output, X);
}

extern "C" __gpu_kernel void LIBM_KERNEL_NAME(LIBM_NAME, Throughput1)(
    uint32_t PassCount, uint32_t ZeroMask, const ArgType *X,
    StorageType *Checksums, uint64_t Count) {
  gpu_libm_bench::measureThroughput<LIBM_SYMBOL, 1>(PassCount, ZeroMask, Count,
                                                    Checksums, X);
}

extern "C" __gpu_kernel void LIBM_KERNEL_NAME(LIBM_NAME, Throughput2)(
    uint32_t PassCount, uint32_t ZeroMask, const ArgType *X,
    StorageType *Checksums, uint64_t Count) {
  gpu_libm_bench::measureThroughput<LIBM_SYMBOL, 2>(PassCount, ZeroMask, Count,
                                                    Checksums, X);
}

extern "C" __gpu_kernel void LIBM_KERNEL_NAME(LIBM_NAME, Throughput4)(
    uint32_t PassCount, uint32_t ZeroMask, const ArgType *X,
    StorageType *Checksums, uint64_t Count) {
  gpu_libm_bench::measureThroughput<LIBM_SYMBOL, 4>(PassCount, ZeroMask, Count,
                                                    Checksums, X);
}

extern "C" __gpu_kernel void LIBM_KERNEL_NAME(LIBM_NAME, Throughput8)(
    uint32_t PassCount, uint32_t ZeroMask, const ArgType *X,
    StorageType *Checksums, uint64_t Count) {
  gpu_libm_bench::measureThroughput<LIBM_SYMBOL, 8>(PassCount, ZeroMask, Count,
                                                    Checksums, X);
}

} // namespace gpu_libm_bench::libm::LIBM_NAME

#endif // LIBM_NAME
#endif // GPU_LIBM_BENCH_KERNELS_LIBM_UNARY_KERNELS_INC
