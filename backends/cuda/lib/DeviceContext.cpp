// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Implements CUDA device information and execution contexts.

#include "DeviceContext.hpp"

#include <cuda.h>

#include <string>
#include <utility>

using namespace gpu_libm_bench;

static Error check(CUresult Result, const char *Operation) {
  if (Result == CUDA_SUCCESS)
    return Error::success();

  const char *Name = nullptr;
  const char *Description = nullptr;
  if (cuGetErrorName(Result, &Name) != CUDA_SUCCESS)
    Name = nullptr;
  if (cuGetErrorString(Result, &Description) != CUDA_SUCCESS)
    Description = nullptr;
  return Error{
      ErrorKind::Runtime,
      std::string(Operation) + ": " + (Name ? Name : "unknown CUDA error") +
          " (" + std::to_string(static_cast<int>(Result)) +
          "): " + (Description ? Description : "no description available")};
}

Expected<int> gpu_libm_bench::getDeviceCount() {
  CUresult Result = cuInit(0);
  if (Result == CUDA_ERROR_NO_DEVICE)
    return 0;
  if (auto Err = check(Result, "cuInit"))
    return Err;

  int Count = 0;
  if (auto Err = check(cuDeviceGetCount(&Count), "cuDeviceGetCount"))
    return Err;
  return Count;
}

Expected<DeviceInfo> gpu_libm_bench::getDeviceInfo(int DeviceID) {
  if (DeviceID < 0)
    return Error{ErrorKind::InvalidArgument, "Device ID must be nonnegative."};

  auto Count = getDeviceCount();
  if (!Count)
    return std::move(Count).error();
  if (DeviceID >= *Count)
    return Error{ErrorKind::InvalidArgument,
                 "Device ID " + std::to_string(DeviceID) +
                     " is out of range; available device count is " +
                     std::to_string(*Count) + "."};

  CUdevice Device;
  if (auto Err = check(cuDeviceGet(&Device, DeviceID), "cuDeviceGet"))
    return Err;

  char Name[256] = {};
  if (auto Err =
          check(cuDeviceGetName(Name, sizeof(Name), Device), "cuDeviceGetName"))
    return Err;

  int Major = 0;
  int Minor = 0;
  if (auto Err = check(
          cuDeviceGetAttribute(
              &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, Device),
          "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)"))
    return Err;
  if (auto Err = check(
          cuDeviceGetAttribute(
              &Minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, Device),
          "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)"))
    return Err;

  std::size_t TotalMemoryBytes = 0;
  if (auto Err = check(cuDeviceTotalMem(&TotalMemoryBytes, Device),
                       "cuDeviceTotalMem"))
    return Err;

  return DeviceInfo{DeviceID, Name,
                    "sm_" + std::to_string(Major) + std::to_string(Minor),
                    TotalMemoryBytes};
}
