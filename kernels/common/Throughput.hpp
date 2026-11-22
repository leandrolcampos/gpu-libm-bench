// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Throughput kernels for independent evaluations of GPU math functions.

#ifndef GPU_LIBM_BENCH_KERNELS_COMMON_THROUGHPUT_HPP
#define GPU_LIBM_BENCH_KERNELS_COMMON_THROUGHPUT_HPP

#include "FPUtil.hpp"
#include "GPU.hpp"

namespace gpu_libm_bench {

template <auto Function, unsigned UnrollFactor, typename StorageType,
          typename... InputTypes>
void measureThroughput(uint64_t Count, StorageType *Checksums,
                       const InputTypes *...Inputs) {
  static_assert(UnrollFactor > 0, "unroll factor must be positive");
  static_assert(sizeof...(InputTypes) > 0, "at least one input is required");
  using FPBits = fputil::FPBits<decltype(Function(Inputs[0]...))>;
  static_assert(__is_same(StorageType, typename FPBits::StorageType),
                "checksum storage type must match the return type");

  const uint64_t ThreadId = gpu::globalThreadId();
  const uint64_t Stride = gpu::gridSize() * gpu::blockSize();
  StorageType Accumulators[UnrollFactor] = {};
  uint64_t Index = ThreadId;

#pragma clang loop unroll(disable)
  for (; Index < Count && Count - Index > (UnrollFactor - 1) * Stride;
       Index += UnrollFactor * Stride) {
#pragma clang loop unroll(full)
    for (unsigned Slot = 0; Slot < UnrollFactor; ++Slot) {
      const uint64_t Offset = Index + Slot * Stride;
      Accumulators[Slot] ^= FPBits(Function(Inputs[Offset]...)).asUInt();
    }
  }

  StorageType Checksum = 0;
#pragma clang loop unroll(full)
  for (unsigned Slot = 0; Slot < UnrollFactor; ++Slot)
    Checksum ^= Accumulators[Slot];

#pragma clang loop unroll(disable)
  for (; Index < Count; Index += Stride)
    Checksum ^= FPBits(Function(Inputs[Index]...)).asUInt();
  Checksums[ThreadId] = Checksum;
}

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_THROUGHPUT_HPP
