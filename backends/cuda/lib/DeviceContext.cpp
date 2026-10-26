// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Implements CUDA device information and execution contexts.

#include "DeviceContext.hpp"

#include <cuda.h>

#include <stdexcept>
#include <string>

using namespace gpu_libm_bench;

static void check(CUresult Result, const char *Operation) {
  if (Result == CUDA_SUCCESS)
    return;

  const char *Name = nullptr;
  const char *Description = nullptr;
  cuGetErrorName(Result, &Name);
  cuGetErrorString(Result, &Description);
  throw std::runtime_error(
      std::string(Operation) + ": " + (Name ? Name : "unknown CUDA error") +
      " (" + std::to_string(static_cast<int>(Result)) +
      "): " + (Description ? Description : "no description available"));
}

int gpu_libm_bench::getDeviceCount() {
  CUresult Result = cuInit(0);
  if (Result == CUDA_ERROR_NO_DEVICE)
    return 0;
  check(Result, "cuInit");

  int Count = 0;
  check(cuDeviceGetCount(&Count), "cuDeviceGetCount");
  return Count;
}

DeviceInfo gpu_libm_bench::getDeviceInfo(int DeviceID) {
  if (DeviceID < 0)
    throw std::invalid_argument("Device ID must be nonnegative.");

  int Count = getDeviceCount();
  if (DeviceID >= Count)
    throw std::invalid_argument("Device ID " + std::to_string(DeviceID) +
                                " is out of range; available device count is " +
                                std::to_string(Count) + ".");

  CUdevice Device;
  check(cuDeviceGet(&Device, DeviceID), "cuDeviceGet");

  char Name[256] = {};
  check(cuDeviceGetName(Name, sizeof(Name), Device), "cuDeviceGetName");

  int Major = 0;
  int Minor = 0;
  check(cuDeviceGetAttribute(
            &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, Device),
        "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)");
  check(cuDeviceGetAttribute(
            &Minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, Device),
        "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)");

  std::size_t TotalMemoryBytes = 0;
  check(cuDeviceTotalMem(&TotalMemoryBytes, Device), "cuDeviceTotalMem");

  return {DeviceID, Name, "sm_" + std::to_string(Major) + std::to_string(Minor),
          TotalMemoryBytes};
}
