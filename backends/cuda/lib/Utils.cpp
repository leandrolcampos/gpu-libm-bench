// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Implements CUDA device state, context activation, and diagnostics.

#include "Utils.hpp"
#include "Context.hpp"

#include <cstdio>

namespace gpu_libm_bench::cuda {

Error check(CUresult Result, const char *Operation) {
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

void reportCleanupError(const Error &Failure) noexcept {
  if (Failure)
    std::fprintf(stderr, "gpu-libm-bench cleanup: %s\n",
                 Failure.Message.c_str());
}

detail::CurrentContextGuard::~CurrentContextGuard() noexcept {
  if (Active) {
    CUcontext PoppedContext = nullptr;
    reportCleanupError(
        check(cuCtxPopCurrent(&PoppedContext), "cuCtxPopCurrent"));
  }
}

detail::CurrentContextGuard::CurrentContextGuard(
    CurrentContextGuard &&Other) noexcept
    : Active(std::exchange(Other.Active, false)) {}

Expected<detail::CurrentContextGuard>
detail::CurrentContextGuard::create(CUcontext Context) {
  if (auto Err = check(cuCtxPushCurrent(Context), "cuCtxPushCurrent"))
    return Err;
  return CurrentContextGuard();
}

detail::DeviceStateTy::~DeviceStateTy() noexcept {
  reportCleanupError(
      check(cuDevicePrimaryCtxRelease(Device), "cuDevicePrimaryCtxRelease"));
}

Expected<std::shared_ptr<detail::DeviceStateTy>>
detail::DeviceStateTy::create(int DeviceID) {
  auto DeviceInfo = getDeviceInfo(DeviceID);
  if (!DeviceInfo)
    return std::move(DeviceInfo).error();

  CUdevice Device;
  if (auto Err = check(cuDeviceGet(&Device, DeviceID), "cuDeviceGet"))
    return Err;

  CUcontext Context = nullptr;
  if (auto Err = check(cuDevicePrimaryCtxRetain(&Context, Device),
                       "cuDevicePrimaryCtxRetain"))
    return Err;

  return std::shared_ptr<DeviceStateTy>(new DeviceStateTy(Device, Context));
}

} // namespace gpu_libm_bench::cuda
