// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0

#include "GPU.hpp"

using namespace gpu_libm_bench;

extern "C" __gpu_kernel void blockStream(const volatile uint32_t *Release,
                                         volatile uint32_t *TimedOut,
                                         uint64_t TimeoutTicks) {
  if (gpu::globalThreadId() != 0)
    return;

  *TimedOut = 0;
  uint64_t Start = gpu::readSteadyCounter();

  while (!*Release) {
    if (gpu::readSteadyCounter() - Start >= TimeoutTicks) {
      *TimedOut = 1;
      break;
    }
  }

  gpu::systemMemoryFence();
}

extern "C" __gpu_kernel void readClockCounters(uint64_t *SteadyCounters,
                                               uint64_t *CycleCounters,
                                               uint32_t *ProcessorIDs) {
  if (gpu::threadId() != 0)
    return;

  uint64_t Index = gpu::blockId();

  uint64_t SteadyCounter = gpu::readSteadyCounter();
  uint64_t CycleCounter = gpu::readCycleCounter();
  uint32_t ProcessorID = gpu::processorId();

  SteadyCounters[Index] = SteadyCounter;
  CycleCounters[Index] = CycleCounter;
  ProcessorIDs[Index] = ProcessorID;
}
