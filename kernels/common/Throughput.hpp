// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Throughput measurement wrappers for GPU math functions.

#ifndef GPU_LIBM_BENCH_KERNELS_COMMON_THROUGHPUT_HPP
#define GPU_LIBM_BENCH_KERNELS_COMMON_THROUGHPUT_HPP

#include "FPUtil.hpp"
#include "GPU.hpp"

namespace gpu_libm_bench {

namespace detail {

template <typename T, unsigned Size> class InputGroup {
public:
  InputGroup(const T *Data, uint64_t Stride) {
#pragma clang loop unroll(full)
    for (unsigned Index = 0; Index < Size; ++Index)
      Values[Index] = Data[Index * Stride];
  }

  T operator[](unsigned Index) const { return Values[Index]; }

private:
  T Values[Size];
};

template <typename T> T xorBits(T Value, uint32_t Mask) {
  if constexpr (__is_same(T, int)) {
    const uint32_t ValueUInt = __builtin_bit_cast(uint32_t, Value);
    return __builtin_bit_cast(T, ValueUInt ^ Mask);
  } else {
    using FPBits = fputil::FPBits<T>;
    using StorageType = typename FPBits::StorageType;

    const auto ValueUInt = FPBits(Value).asUInt();
    return FPBits(StorageType(ValueUInt ^ static_cast<StorageType>(Mask)))
        .getValue();
  }
}

template <auto Function, unsigned Size, typename StorageType,
          typename... InputTypes>
void accumulateGroupChecksums(uint32_t PassCount, uint32_t ZeroMask,
                              StorageType (&Checksums)[Size],
                              const InputGroup<InputTypes, Size> &...Inputs) {
  using FPBits = fputil::FPBits<decltype(Function(Inputs[0]...))>;
  static_assert(__is_same(StorageType, typename FPBits::StorageType),
                "checksum storage type must match the return type");

  StorageType Sums[Size] = {};
#pragma clang loop unroll(disable)
  for (uint32_t Pass = 0; Pass < PassCount; ++Pass) {
    // The runtime zero mask preserves the input and keeps the evaluations
    // inside the loop after device linking and assembly.
    const uint32_t PassMask = Pass & ZeroMask;

#pragma clang loop unroll(full)
    for (unsigned Index = 0; Index < Size; ++Index)
      Sums[Index] +=
          FPBits(Function(xorBits(Inputs[Index], PassMask)...)).asUInt();
  }

#pragma clang loop unroll(full)
  for (unsigned Index = 0; Index < Size; ++Index)
    Checksums[Index] ^= Sums[Index];
}

} // namespace detail

template <auto Function, unsigned UnrollFactor, typename StorageType,
          typename... InputTypes>
void measureThroughput(uint32_t PassCount, uint32_t ZeroMask, uint64_t Count,
                       StorageType *Checksums, const InputTypes *...Inputs) {
  static_assert(UnrollFactor > 0, "unroll factor must be positive");
  static_assert(sizeof...(InputTypes) > 0, "at least one input is required");

  const uint64_t ThreadId = gpu::globalThreadId();
  const uint64_t Stride = gpu::gridSize() * gpu::blockSize();
  StorageType Accumulators[UnrollFactor] = {};
  uint64_t Index = ThreadId;

#pragma clang loop unroll(disable)
  for (; Index < Count && Count - Index > (UnrollFactor - 1) * Stride;
       Index += UnrollFactor * Stride)
    detail::accumulateGroupChecksums<Function>(
        PassCount, ZeroMask, Accumulators,
        detail::InputGroup<InputTypes, UnrollFactor>(Inputs + Index,
                                                     Stride)...);

  StorageType Checksum[1] = {};
#pragma clang loop unroll(full)
  for (unsigned Slot = 0; Slot < UnrollFactor; ++Slot)
    Checksum[0] ^= Accumulators[Slot];

#pragma clang loop unroll(disable)
  for (; Index < Count; Index += Stride)
    detail::accumulateGroupChecksums<Function>(
        PassCount, ZeroMask, Checksum,
        detail::InputGroup<InputTypes, 1>(Inputs + Index, Stride)...);

  Checksums[ThreadId] = Checksum[0];
}

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_THROUGHPUT_HPP
