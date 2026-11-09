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

// Linear thread and block indices, with x varying fastest.

inline uint64_t threadId() {
  return __gpu_thread_id_x() +
         static_cast<uint64_t>(__gpu_num_threads_x()) *
             (__gpu_thread_id_y() +
              static_cast<uint64_t>(__gpu_num_threads_y()) *
                  __gpu_thread_id_z());
}

inline uint64_t blockId() {
  return __gpu_block_id_x() +
         static_cast<uint64_t>(__gpu_num_blocks_x()) *
             (__gpu_block_id_y() +
              static_cast<uint64_t>(__gpu_num_blocks_y()) * __gpu_block_id_z());
}

inline uint64_t globalThreadId() {
  return blockId() * blockSize() + threadId();
}

// Fixed-frequency timer. Units and frequency are backend-specific.

inline uint64_t readTimer() { return __builtin_readsteadycounter(); }

} // namespace gpu_libm_bench::gpu

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_GPU_HPP
