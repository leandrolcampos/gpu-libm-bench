// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Device timer validation kernels.

#include "GPU.hpp"

extern "C" __gpu_kernel void timerReadings(uint64_t *Readings, uint32_t Count) {
  if (gpu_libm_bench::gpu::globalThreadId() != 0)
    return;

#pragma clang loop unroll(disable)
  for (uint32_t Index = 0; Index < Count; ++Index) {
    uint64_t Start = gpu_libm_bench::gpu::readTimer();
    uint64_t Stop = gpu_libm_bench::gpu::readTimer();
    Readings[2 * static_cast<uint64_t>(Index)] = Start;
    Readings[2 * static_cast<uint64_t>(Index) + 1] = Stop;
  }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
extern "C" __gpu_kernel void timerWork(uint64_t *Elapsed, uint32_t *Outputs,
                                       uint32_t Iterations, uint32_t Count,
                                       uint32_t Seed, uint32_t Mask) {
  if (gpu_libm_bench::gpu::globalThreadId() != 0)
    return;

#pragma clang loop unroll(disable)
  for (uint32_t Index = 0; Index < Count; ++Index) {
    uint32_t Value = Seed;
    uint64_t Start = gpu_libm_bench::gpu::readTimer();
    // Mask must be zero at runtime, while remaining unknown to the compiler.
    Value ^= static_cast<uint32_t>(Start) & Mask;

    // The loop exit depends on completion of the last arithmetic result.
#pragma clang loop unroll(disable)
    for (uint32_t Step = 0; Step < Iterations;) {
      Value = Value * 1664525U + 1013904223U;
      Step = (Step + 1) ^ (Value & Mask);
    }

    uint64_t Stop = gpu_libm_bench::gpu::readTimer();

    Elapsed[Index] = Stop - Start;
    Outputs[Index] = Value;
  }
}
// NOLINTEND(bugprone-easily-swappable-parameters)
