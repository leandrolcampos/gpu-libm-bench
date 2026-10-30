// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Floating-point representation utilities for GPU kernels.

#ifndef GPU_LIBM_BENCH_KERNELS_COMMON_FPUTIL_HPP
#define GPU_LIBM_BENCH_KERNELS_COMMON_FPUTIL_HPP

#include <stdint.h>

namespace gpu_libm_bench::fputil {

namespace detail {

template <typename FloatType> struct FPStorageType;

template <> struct FPStorageType<_Float16> {
  using Type = uint16_t;
};

template <> struct FPStorageType<float> {
  using Type = uint32_t;
};

template <> struct FPStorageType<double> {
  using Type = uint64_t;
};

} // namespace detail

template <typename FloatType> class FPBits {
public:
  using StorageType = typename detail::FPStorageType<FloatType>::Type;

  constexpr explicit FPBits(FloatType Value)
      : Bits(__builtin_bit_cast(StorageType, Value)) {}

  constexpr explicit FPBits(StorageType Bits) : Bits(Bits) {}

  template <typename ValueType> explicit FPBits(ValueType) = delete;

  constexpr FloatType getValue() const {
    return __builtin_bit_cast(FloatType, Bits);
  }

  constexpr StorageType asUInt() const { return Bits; }

private:
  StorageType Bits;
};

FPBits(uint16_t) -> FPBits<_Float16>;
FPBits(uint32_t) -> FPBits<float>;
FPBits(uint64_t) -> FPBits<double>;

} // namespace gpu_libm_bench::fputil

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_FPUTIL_HPP
