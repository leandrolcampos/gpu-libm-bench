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
// Kernel entry points for unary math functions.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_KERNELS_LIBM_UNARY_KERNELS_INC
#define GPU_LIBM_BENCH_KERNELS_LIBM_UNARY_KERNELS_INC

#ifdef LIBM_FUNCTION_NAME
#include "Evaluation.hpp"
#include "Latency.hpp"
#include "Macros.hpp"
#include "Throughput.hpp"

namespace gpu_libm_bench::libm::LIBM_FUNCTION_NAME {

using ArgType = LIBM_FUNCTION_ARG_TYPE;
using ReturnType = LIBM_FUNCTION_RETURN_TYPE;
using StorageType = gpu_libm_bench::fputil::FPBits<ReturnType>::StorageType;

extern "C" ReturnType LIBM_FUNCTION_SYMBOL(ArgType);

extern "C" __gpu_kernel void LIBM_KERNEL_NAME(LIBM_FUNCTION_NAME,
                                              Evaluation)(uint64_t Count,
                                                          ReturnType *Output,
                                                          const ArgType *X) {
  gpu_libm_bench::evaluate<LIBM_FUNCTION_SYMBOL>(Count, Output, X);
}

extern "C" __gpu_kernel void LIBM_KERNEL_NAME(LIBM_FUNCTION_NAME, Latency)(
    uint32_t PassCount, uint32_t ChainLength, const StorageType *ZeroMask,
    uint64_t *ElapsedTicks, StorageType *Output, const ArgType *X) {
  gpu_libm_bench::measureLatency<LIBM_FUNCTION_SYMBOL>(
      PassCount, ChainLength, ZeroMask, ElapsedTicks, Output, X);
}

extern "C" __gpu_kernel void LIBM_KERNEL_NAME(LIBM_FUNCTION_NAME, Throughput1)(
    uint32_t PassCount, uint32_t ZeroMask, uint64_t Count,
    StorageType *Checksums, const ArgType *X) {
  gpu_libm_bench::measureThroughput<LIBM_FUNCTION_SYMBOL, 1>(
      PassCount, ZeroMask, Count, Checksums, X);
}

extern "C" __gpu_kernel void LIBM_KERNEL_NAME(LIBM_FUNCTION_NAME, Throughput2)(
    uint32_t PassCount, uint32_t ZeroMask, uint64_t Count,
    StorageType *Checksums, const ArgType *X) {
  gpu_libm_bench::measureThroughput<LIBM_FUNCTION_SYMBOL, 2>(
      PassCount, ZeroMask, Count, Checksums, X);
}

extern "C" __gpu_kernel void LIBM_KERNEL_NAME(LIBM_FUNCTION_NAME, Throughput4)(
    uint32_t PassCount, uint32_t ZeroMask, uint64_t Count,
    StorageType *Checksums, const ArgType *X) {
  gpu_libm_bench::measureThroughput<LIBM_FUNCTION_SYMBOL, 4>(
      PassCount, ZeroMask, Count, Checksums, X);
}

extern "C" __gpu_kernel void LIBM_KERNEL_NAME(LIBM_FUNCTION_NAME, Throughput8)(
    uint32_t PassCount, uint32_t ZeroMask, uint64_t Count,
    StorageType *Checksums, const ArgType *X) {
  gpu_libm_bench::measureThroughput<LIBM_FUNCTION_SYMBOL, 8>(
      PassCount, ZeroMask, Count, Checksums, X);
}

} // namespace gpu_libm_bench::libm::LIBM_FUNCTION_NAME

#endif // LIBM_FUNCTION_NAME
#endif // GPU_LIBM_BENCH_KERNELS_LIBM_UNARY_KERNELS_INC
