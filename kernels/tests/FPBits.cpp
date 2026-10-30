// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Compile-time checks for floating-point representations.

#include "FPUtil.hpp"

template <typename FloatType, typename UIntType, unsigned Count>
constexpr bool checkFPBits(const UIntType (&Patterns)[Count]) {
  using FPBits = gpu_libm_bench::fputil::FPBits<FloatType>;

  static_assert(__is_same(typename FPBits::StorageType, UIntType));
  static_assert(
      __is_same(decltype(gpu_libm_bench::fputil::FPBits(UIntType{})), FPBits));
  static_assert(sizeof(FPBits) == sizeof(UIntType));
  static_assert(!__is_constructible(FPBits, int));
  static_assert(!__is_constructible(FPBits, long double));

  for (UIntType Pattern : Patterns)
    if (FPBits(FPBits(Pattern).getValue()).asUInt() != Pattern)
      return false;

  return true;
}

// Finite normals, signed zeros, subnormals, infinities, and NaNs with payloads.
constexpr uint16_t Float16Patterns[] = {0x3e00, 0xc180, 0x0000, 0x8000, 0x0001,
                                        0x8001, 0x7c00, 0xfc00, 0x7e35, 0x7c35};
constexpr uint32_t FloatPatterns[] = {
    0x3fc00000, 0xc0300000, 0x00000000, 0x80000000, 0x00000001,
    0x80000001, 0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345};
constexpr uint64_t DoublePatterns[] = {0x3ff8000000000000, 0xc006000000000000,
                                       0x0000000000000000, 0x8000000000000000,
                                       0x0000000000000001, 0x8000000000000001,
                                       0x7ff0000000000000, 0xfff0000000000000,
                                       0x7ff8123456789abc, 0x7ff0123456789abc};

static_assert(checkFPBits<_Float16>(Float16Patterns));
static_assert(checkFPBits<float>(FloatPatterns));
static_assert(checkFPBits<double>(DoublePatterns));
