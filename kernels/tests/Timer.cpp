// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Device timer validation kernels.

#include "GPU.hpp"

extern "C" __gpu_kernel void timerReadings(uint64_t *Readings, uint32_t Count) {
  if (gpu_libm_bench::gpu::globalThreadId() != 0)
    return;

#pragma clang loop unroll(disable)
  for (uint32_t Index = 0; Index < Count; ++Index) {
    uint64_t Start = gpu_libm_bench::gpu::readSteadyCounter();
    uint64_t Stop = gpu_libm_bench::gpu::readSteadyCounter();

    Readings[2 * static_cast<uint64_t>(Index)] = Start;
    Readings[2 * static_cast<uint64_t>(Index) + 1] = Stop;
  }
}
