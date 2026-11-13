// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Device execution primitives for GPU kernels.

#ifndef GPU_LIBM_BENCH_KERNELS_COMMON_GPU_HPP
#define GPU_LIBM_BENCH_KERNELS_COMMON_GPU_HPP

#include <gpuintrin.h>

namespace gpu_libm_bench::gpu {

// Launch dimensions.

inline uint64_t blockSize() {
  return static_cast<uint64_t>(__gpu_num_threads_x()) * __gpu_num_threads_y() *
         __gpu_num_threads_z();
}

// Linear block and thread indices, with x varying fastest.

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

// Processor identification.

inline uint32_t processorId() {
#if defined(__NVPTX__)
  return __nvvm_read_ptx_sreg_smid();
#else
#error "processorId is not implemented for this GPU target"
#endif
}

// Clock counters.

inline uint64_t readCycleCounter() { return __builtin_readcyclecounter(); }

inline uint64_t readSteadyCounter() { return __builtin_readsteadycounter(); }

// Memory fences.

inline void systemMemoryFence() {
  __scoped_atomic_thread_fence(__ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM);
}

} // namespace gpu_libm_bench::gpu

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_GPU_HPP
