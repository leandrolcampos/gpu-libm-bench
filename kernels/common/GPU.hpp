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
// Device execution primitives for GPU kernels.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_KERNELS_COMMON_GPU_HPP
#define GPU_LIBM_BENCH_KERNELS_COMMON_GPU_HPP

#include <gpuintrin.h>

#include <stdint.h>

namespace gpu_libm_bench::gpu {

//===----------------------------------------------------------------------===//
// Launch dimensions
//===----------------------------------------------------------------------===//

inline uint64_t gridSize() {
  return static_cast<uint64_t>(__gpu_num_blocks_x()) * __gpu_num_blocks_y() *
         __gpu_num_blocks_z();
}

inline uint64_t blockSize() {
  return static_cast<uint64_t>(__gpu_num_threads_x()) * __gpu_num_threads_y() *
         __gpu_num_threads_z();
}

//===----------------------------------------------------------------------===//
// Linear block and thread indices
//===----------------------------------------------------------------------===//

inline uint64_t blockId() {
  return __gpu_block_id_x() +
         static_cast<uint64_t>(__gpu_num_blocks_x()) *
             (__gpu_block_id_y() +
              static_cast<uint64_t>(__gpu_num_blocks_y()) * __gpu_block_id_z());
}

inline uint64_t threadId() {
  return __gpu_thread_id_x() +
         static_cast<uint64_t>(__gpu_num_threads_x()) *
             (__gpu_thread_id_y() +
              static_cast<uint64_t>(__gpu_num_threads_y()) *
                  __gpu_thread_id_z());
}

inline uint64_t globalThreadId() {
  return blockId() * blockSize() + threadId();
}

//===----------------------------------------------------------------------===//
// Hardware identifier of the current multiprocessor
//===----------------------------------------------------------------------===//

// Returned IDs is not guaranteed to be contiguous.
inline uint32_t multiprocessorId() {
#if defined(__AMDGCN__)

  // AMD ISA register layouts and ROCm HIP's __smid() are references for the
  // fields used here.

#if defined(__GFX6__) || defined(__GFX7__) || defined(__GFX8__) ||             \
    defined(__GFX9__)
  uint32_t HardwareId;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 8)" : "=s"(HardwareId));

#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx9_4_generic__)
  uint32_t XccId;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 4)" : "=s"(XccId));
  HardwareId |= XccId << 8U;
#endif // (__gfx942__ || __gfx950__ || __gfx9_4_generic__)

  return HardwareId;

#elif defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
  uint32_t HardwareId;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(HardwareId));

#if defined(__GFX12__)
  constexpr unsigned GetShaderEngineHardwareId = 0x87;
  const uint32_t EngineHardwareId =
      __builtin_amdgcn_s_sendmsg_rtn(GetShaderEngineHardwareId);
  const uint32_t Engine = EngineHardwareId & 0xfU;

#if defined(__gfx1250__) || defined(__gfx1250_strict__) ||                     \
    defined(__gfx1251__) || defined(__gfx12_5_generic__)
  const uint32_t Die = (EngineHardwareId >> 16U) & 0xfU;
#else
  const uint32_t Die = (EngineHardwareId >> 8U) & 0xfU;
#endif // CDNA 5

#elif defined(__GFX10__)
  const uint32_t Engine = (HardwareId >> 18U) & 3U;
  constexpr uint32_t Die = 0;
#else
  const uint32_t Engine = (HardwareId >> 18U) & 7U;
  constexpr uint32_t Die = 0;
#endif // __GFX12__

  const uint32_t Array = (HardwareId >> 16U) & 1U;
  const uint32_t WorkgroupProcessor = (HardwareId >> 10U) & 0xfU;
  const uint32_t Id =
      (Die << 9U) | (Engine << 5U) | (Array << 4U) | WorkgroupProcessor;

  if constexpr (__AMDGCN_CUMODE__)
    return (Id << 1U) | ((HardwareId >> 8U) & 1U);
  return Id;

#else
#error "unsupported AMD GPU architecture"
#endif // AMD architecture selection

#elif defined(__NVPTX__)
  return __nvvm_read_ptx_sreg_smid();
#else
#error "unsupported GPU target"
#endif // __AMDGCN__
}

//===----------------------------------------------------------------------===//
// Clock counters
//===----------------------------------------------------------------------===//

inline uint64_t readSteadyCounter() { return __builtin_readsteadycounter(); }

inline uint64_t readCycleCounter() { return __builtin_readcyclecounter(); }

//===----------------------------------------------------------------------===//
// Atomics and memory fences
//===----------------------------------------------------------------------===//

enum class MemoryOrder : uint8_t {
  Relaxed = __ATOMIC_RELAXED,
  Acquire = __ATOMIC_ACQUIRE,
  Release = __ATOMIC_RELEASE,
  AcqRel = __ATOMIC_ACQ_REL,
  SeqCst = __ATOMIC_SEQ_CST
};

enum class MemoryScope : uint8_t {
  Block = __MEMORY_SCOPE_WRKGRP,
  Device = __MEMORY_SCOPE_DEVICE,
  System = __MEMORY_SCOPE_SYSTEM,
};

template <MemoryOrder Order = MemoryOrder::SeqCst,
          MemoryScope Scope = MemoryScope::System, typename T>
inline T atomicExchange(T *Object, T Desired) {
  return __scoped_atomic_exchange_n(Object, Desired, static_cast<int>(Order),
                                    static_cast<int>(Scope));
}

template <MemoryOrder Order = MemoryOrder::SeqCst,
          MemoryScope Scope = MemoryScope::System>
inline void atomicThreadFence() {
  __scoped_atomic_thread_fence(static_cast<int>(Order),
                               static_cast<int>(Scope));
}

} // namespace gpu_libm_bench::gpu

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_GPU_HPP
