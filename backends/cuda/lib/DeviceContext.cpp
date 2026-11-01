// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Implements CUDA device information and execution contexts.

#include "DeviceContext.hpp"
#include "Utils.hpp"

namespace gpu_libm_bench::cuda {

Expected<int> getDeviceCount() {
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

Expected<DeviceInfo> getDeviceInfo(int DeviceID) {
  if (DeviceID < 0)
    return Error{ErrorKind::InvalidArgument, "device ID must be nonnegative"};

  auto Count = getDeviceCount();
  if (!Count)
    return std::move(Count).error();
  if (DeviceID >= *Count)
    return Error{ErrorKind::InvalidArgument,
                 "device ID " + std::to_string(DeviceID) +
                     " is out of range; available device count is " +
                     std::to_string(*Count)};

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

  // %globaltimer reports nanoseconds, independently of the GPU core clock.
  return DeviceInfo{DeviceID,
                    Name,
                    "sm_" + std::to_string(Major) + std::to_string(Minor),
                    TotalMemoryBytes,
                    {"%globaltimer", 64, 1'000'000'000}};
}

DeviceContext::DeviceContext(std::shared_ptr<detail::DeviceState> Device,
                             void *Stream)
    : Device(std::move(Device)), Stream(Stream) {}

DeviceContext::DeviceContext(DeviceContext &&Other) noexcept
    : Device(std::move(Other.Device)),
      Stream(std::exchange(Other.Stream, nullptr)) {}

DeviceContext::~DeviceContext() noexcept { reportCleanupError(close()); }

Expected<DeviceContext> DeviceContext::create(int DeviceID) {
  auto Device = detail::DeviceState::create(DeviceID);
  if (!Device)
    return std::move(Device).error();

  auto Guard = detail::CurrentContextGuard::create((*Device)->getContext());
  if (!Guard)
    return std::move(Guard).error();

  CUstream Stream = nullptr;
  if (auto Err = check(cuStreamCreate(&Stream, CU_STREAM_NON_BLOCKING),
                       "cuStreamCreate"))
    return Err;

  return DeviceContext(*std::move(Device), Stream);
}

Error DeviceContext::ensureOpen() const {
  if (isClosed())
    return Error{ErrorKind::Runtime, "device context is closed"};
  return Error::success();
}

Error DeviceContext::validateBuffer(const DeviceBuffer &Buffer) const {
  if (auto Err = ensureOpen())
    return Err;
  if (!Buffer.Device)
    return Error{ErrorKind::InvalidArgument, "device buffer is invalid"};
  if (Buffer.Device->getContext() != Device->getContext())
    return Error{ErrorKind::InvalidArgument,
                 "device buffer belongs to a different CUDA device"};
  return Error::success();
}

Error DeviceContext::validateKernel(const DeviceKernel &Kernel) const {
  if (auto Err = ensureOpen())
    return Err;
  if (!Kernel.Module)
    return Error{ErrorKind::InvalidArgument, "kernel is invalid"};
  if (Kernel.getDevice()->getContext() != Device->getContext())
    return Error{ErrorKind::InvalidArgument,
                 "kernel belongs to a different CUDA device"};
  return Error::success();
}

Error DeviceContext::close() {
  if (isClosed())
    return Error::success();

  Error Failure;
  {
    auto Guard = detail::CurrentContextGuard::create(Device->getContext());
    if (!Guard)
      return std::move(Guard).error();

    Failure = check(cuStreamSynchronize(static_cast<CUstream>(Stream)),
                    "cuStreamSynchronize");

    auto DestroyFailure = check(cuStreamDestroy(static_cast<CUstream>(Stream)),
                                "cuStreamDestroy");
    if (DestroyFailure) {
      if (Failure)
        Failure.Message += "; " + DestroyFailure.Message;
      else
        Failure = std::move(DestroyFailure);
    } else {
      Stream = nullptr;
    }
  }

  if (!Stream)
    Device.reset();

  return Failure;
}

Expected<DeviceBuffer>
DeviceContext::createBuffer(std::size_t ByteCount) const {
  if (auto Err = ensureOpen())
    return Err;
  return DeviceBuffer::create(Device, ByteCount);
}

Expected<DeviceModule>
DeviceContext::loadModule(const std::string &Path) const {
  if (auto Err = ensureOpen())
    return Err;
  return DeviceModule::load(Device, Path);
}

Error DeviceContext::copyToDevice(DeviceBuffer &Destination, const void *Source,
                                  std::size_t ByteCount) const {
  if (auto Err = validateBuffer(Destination))
    return Err;
  if (ByteCount > Destination.getByteCount())
    return Error{ErrorKind::InvalidArgument,
                 "copy exceeds destination device buffer capacity"};
  if (!ByteCount)
    return Error::success();
  if (!Source)
    return Error{ErrorKind::InvalidArgument, "copy source is null"};

  auto Guard = detail::CurrentContextGuard::create(Device->getContext());
  if (!Guard)
    return std::move(Guard).error();

  if (auto Err = check(cuMemcpyHtoDAsync(Destination.Pointer, Source, ByteCount,
                                         static_cast<CUstream>(Stream)),
                       "cuMemcpyHtoDAsync"))
    return Err;

  return check(cuStreamSynchronize(static_cast<CUstream>(Stream)),
               "cuStreamSynchronize");
}

Error DeviceContext::copyToHost(void *Destination, const DeviceBuffer &Source,
                                std::size_t ByteCount) const {
  if (auto Err = validateBuffer(Source))
    return Err;
  if (ByteCount > Source.getByteCount())
    return Error{ErrorKind::InvalidArgument,
                 "copy exceeds source device buffer capacity"};
  if (!ByteCount)
    return Error::success();
  if (!Destination)
    return Error{ErrorKind::InvalidArgument, "copy destination is null"};

  auto Guard = detail::CurrentContextGuard::create(Device->getContext());
  if (!Guard)
    return std::move(Guard).error();

  if (auto Err = check(cuMemcpyDtoHAsync(Destination, Source.Pointer, ByteCount,
                                         static_cast<CUstream>(Stream)),
                       "cuMemcpyDtoHAsync"))
    return Err;

  return check(cuStreamSynchronize(static_cast<CUstream>(Stream)),
               "cuStreamSynchronize");
}

Error DeviceContext::launch(const DeviceKernel &Kernel,
                            const std::vector<LaunchArgument> &Arguments,
                            const LaunchConfig &Config) const {
  if (auto Err = validateKernel(Kernel))
    return Err;
  if (!Config.isValid())
    return Error{ErrorKind::InvalidArgument,
                 "invalid kernel launch configuration"};

  std::vector<CUdeviceptr> DevicePointers(Arguments.size());
  std::vector<void *> Pointers;
  Pointers.reserve(Arguments.size());

  for (std::size_t Index = 0; Index < Arguments.size(); ++Index) {
    if (const auto *Buffer =
            std::get_if<const DeviceBuffer *>(&Arguments[Index])) {
      if (!*Buffer)
        return Error{ErrorKind::InvalidArgument,
                     "kernel argument has a null buffer"};
      if (auto Err = validateBuffer(**Buffer))
        return Err;

      DevicePointers[Index] = (*Buffer)->Pointer;
      Pointers.push_back(&DevicePointers[Index]);
    } else {
      const void *Scalar = std::get<const void *>(Arguments[Index]);
      if (!Scalar)
        return Error{ErrorKind::InvalidArgument,
                     "kernel scalar storage is null"};

      // The driver reads argument values through a non-const C interface.
      Pointers.push_back(const_cast<void *>(Scalar));
    }
  }

  auto Guard = detail::CurrentContextGuard::create(Device->getContext());
  if (!Guard)
    return std::move(Guard).error();

  if (auto Err =
          check(cuLaunchKernel(
                    static_cast<CUfunction>(Kernel.Function), Config.Grid.X,
                    Config.Grid.Y, Config.Grid.Z, Config.Block.X,
                    Config.Block.Y, Config.Block.Z,
                    static_cast<unsigned int>(Config.SharedMemoryBytes),
                    static_cast<CUstream>(Stream), Pointers.data(), nullptr),
                "cuLaunchKernel"))
    return Err;

  return check(cuStreamSynchronize(static_cast<CUstream>(Stream)),
               "cuStreamSynchronize");
}

} // namespace gpu_libm_bench::cuda
