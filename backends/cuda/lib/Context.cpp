// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Implements CUDA device information and execution contexts.

#include "Context.hpp"
#include "KernelProfiler.hpp"
#include "Utils.hpp"

#include <filesystem>

namespace gpu_libm_bench::cuda {

//
// Device queries
//

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

Expected<DeviceInfoTy> getDeviceInfo(int DeviceID) {
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
  return DeviceInfoTy{DeviceID,
                      Name,
                      "sm_" + std::to_string(Major) + std::to_string(Minor),
                      TotalMemoryBytes,
                      {"%globaltimer", 64, 1'000'000'000}};
}

//
// Support image configuration
//

static std::filesystem::path ImagesBuildRoot;

static std::string getSupportImagePath(const std::string &Architecture) {
  if (ImagesBuildRoot.empty())
    return {};

  return (ImagesBuildRoot / "cuda" / Architecture / "support" / "support.cubin")
      .string();
}

Error detail::setImagesBuildRoot(const std::string &Root) {
  std::filesystem::path Path(Root);
  if (Path.empty() || !Path.is_absolute())
    return Error{ErrorKind::InvalidArgument,
                 "kernel images build root must be an absolute path"};

  Path = Path.lexically_normal();
  if (!ImagesBuildRoot.empty() && ImagesBuildRoot != Path)
    return Error{ErrorKind::InvalidArgument,
                 "kernel images build root is already configured"};

  ImagesBuildRoot = std::move(Path);
  return Error::success();
}

//
// Context lifecycle
//

Expected<DeviceContext> DeviceContext::create(int DeviceID) {
  auto DeviceState = detail::DeviceStateTy::create(DeviceID);
  if (!DeviceState)
    return std::move(DeviceState).error();

  auto DeviceInfo = getDeviceInfo(DeviceID);
  if (!DeviceInfo)
    return std::move(DeviceInfo).error();

  auto Guard =
      detail::CurrentContextGuard::create((*DeviceState)->getContext());
  if (!Guard)
    return std::move(Guard).error();

  CUstream Stream = nullptr;
  if (auto Err = check(cuStreamCreate(&Stream, CU_STREAM_NON_BLOCKING),
                       "cuStreamCreate"))
    return Err;

  return DeviceContext(*std::move(DeviceState), Stream,
                       DeviceInfo->Architecture);
}

DeviceContext::DeviceContext(DeviceContext &&Other) noexcept
    : DeviceState(std::move(Other.DeviceState)),
      Stream(std::exchange(Other.Stream, nullptr)),
      SupportImagePath(std::move(Other.SupportImagePath)),
      SupportModule(std::move(Other.SupportModule)),
      Profiler(std::move(Other.Profiler)) {}

DeviceContext::~DeviceContext() noexcept { reportCleanupError(release()); }

Error DeviceContext::release() {
  if (isReleased())
    return Error::success();

  Error Failure;
  {
    auto Guard = detail::CurrentContextGuard::create(DeviceState->getContext());
    if (!Guard)
      return std::move(Guard).error();

    Failure = check(cuStreamSynchronize(static_cast<CUstream>(Stream)),
                    "cuStreamSynchronize");

    // The profiler uses the support module and stream during cleanup.
    Profiler.reset();
    SupportModule.reset();

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
    DeviceState.reset();

  return Failure;
}

//
// Resources and memory transfers
//

Expected<DeviceBuffer>
DeviceContext::createBuffer(std::size_t ByteCount) const {
  if (auto Err = ensureActive())
    return Err;

  return DeviceBuffer::create(DeviceState, ByteCount);
}

Expected<DeviceModule>
DeviceContext::loadModule(const std::string &Path) const {
  if (auto Err = ensureActive())
    return Err;

  return DeviceModule::load(DeviceState, Path);
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

  auto Guard = detail::CurrentContextGuard::create(DeviceState->getContext());
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

  auto Guard = detail::CurrentContextGuard::create(DeviceState->getContext());
  if (!Guard)
    return std::move(Guard).error();

  if (auto Err = check(cuMemcpyDtoHAsync(Destination, Source.Pointer, ByteCount,
                                         static_cast<CUstream>(Stream)),
                       "cuMemcpyDtoHAsync"))
    return Err;

  return check(cuStreamSynchronize(static_cast<CUstream>(Stream)),
               "cuStreamSynchronize");
}

//
// Kernel execution
//

Error DeviceContext::launch(const DeviceKernel &Kernel,
                            const std::vector<KernelArgument> &KernelArguments,
                            const LaunchConfig &Config) const {
  if (auto Err = ensureActive())
    return Err;

  auto Guard = detail::CurrentContextGuard::create(DeviceState->getContext());
  if (!Guard)
    return std::move(Guard).error();

  if (auto Err = enqueueKernel(Kernel, KernelArguments, Config))
    return Err;

  return check(cuStreamSynchronize(static_cast<CUstream>(Stream)),
               "cuStreamSynchronize");
}

Expected<KernelProfile>
DeviceContext::profile(const DeviceKernel &Kernel,
                       const std::vector<KernelArgument> &KernelArguments,
                       const LaunchConfig &Config,
                       const std::set<KernelMetric> &Metrics) const {
  if (auto Err = validateKernel(Kernel))
    return Err;

  auto Guard = detail::CurrentContextGuard::create(DeviceState->getContext());
  if (!Guard)
    return std::move(Guard).error();

  if (!Profiler) {
    if (auto Err = detail::KernelProfiler::validateMetrics(Metrics))
      return Err;

    auto Module = getSupportModule();
    if (!Module)
      return std::move(Module).error();

    auto Created = detail::KernelProfiler::create(
        DeviceState->getContext(), static_cast<CUstream>(Stream), **Module);
    if (!Created)
      return std::move(Created).error();

    Profiler = *std::move(Created);
  }

  return Profiler->profile(
      [&] { return enqueueKernel(Kernel, KernelArguments, Config); }, Metrics);
}

//
// Internal helpers
//

DeviceContext::DeviceContext(std::shared_ptr<detail::DeviceStateTy> DeviceState,
                             void *Stream, const std::string &Architecture)
    : DeviceState(std::move(DeviceState)), Stream(Stream),
      SupportImagePath(getSupportImagePath(Architecture)) {}

Expected<const DeviceModule *> DeviceContext::getSupportModule() const {
  if (auto Err = ensureActive())
    return Err;

  if (!SupportModule) {
    if (SupportImagePath.empty())
      return Error{ErrorKind::Runtime,
                   "kernel images build root is not configured"};

    auto Module = loadModule(SupportImagePath);
    if (!Module)
      return std::move(Module).error();

    SupportModule.emplace(*std::move(Module));
  }

  return &*SupportModule;
}

Error DeviceContext::enqueueKernel(
    const DeviceKernel &Kernel,
    const std::vector<KernelArgument> &KernelArguments,
    const LaunchConfig &Config) const {
  if (auto Err = validateKernel(Kernel))
    return Err;
  if (!Config.isValid())
    return Error{ErrorKind::InvalidArgument,
                 "invalid kernel launch configuration"};
  if (auto Err = validateKernelArguments(KernelArguments))
    return Err;

  // Keep device pointer values alive while the driver copies the arguments.
  std::vector<CUdeviceptr> DevicePointers(KernelArguments.size());
  std::vector<void *> Pointers;
  Pointers.reserve(KernelArguments.size());

  for (std::size_t Index = 0; Index < KernelArguments.size(); ++Index) {
    if (const auto *Buffer =
            std::get_if<const DeviceBuffer *>(&KernelArguments[Index])) {
      DevicePointers[Index] = (*Buffer)->Pointer;
      Pointers.push_back(&DevicePointers[Index]);
    } else {
      const void *Scalar = std::get<const void *>(KernelArguments[Index]);
      // The driver reads argument values through a non-const C interface.
      Pointers.push_back(const_cast<void *>(Scalar));
    }
  }

  return check(
      cuLaunchKernel(static_cast<CUfunction>(Kernel.Function), Config.Grid.X,
                     Config.Grid.Y, Config.Grid.Z, Config.Block.X,
                     Config.Block.Y, Config.Block.Z,
                     static_cast<unsigned int>(Config.SharedMemoryBytes),
                     static_cast<CUstream>(Stream), Pointers.data(), nullptr),
      "cuLaunchKernel");
}

Error DeviceContext::ensureActive() const {
  if (isReleased())
    return Error{ErrorKind::Runtime, "device context has been released"};

  return Error::success();
}

Error DeviceContext::validateBuffer(const DeviceBuffer &Buffer) const {
  if (auto Err = ensureActive())
    return Err;

  if (!Buffer.DeviceState)
    return Error{ErrorKind::InvalidArgument, "device buffer is invalid"};
  if (Buffer.DeviceState->getContext() != DeviceState->getContext())
    return Error{ErrorKind::InvalidArgument,
                 "device buffer belongs to a different CUDA device"};

  return Error::success();
}

Error DeviceContext::validateKernel(const DeviceKernel &Kernel) const {
  if (auto Err = ensureActive())
    return Err;

  if (!Kernel.Module)
    return Error{ErrorKind::InvalidArgument, "kernel is invalid"};
  if (Kernel.getDeviceState()->getContext() != DeviceState->getContext())
    return Error{ErrorKind::InvalidArgument,
                 "kernel belongs to a different CUDA device"};

  return Error::success();
}

Error DeviceContext::validateKernelArguments(
    const std::vector<KernelArgument> &KernelArguments) const {
  for (const auto &Argument : KernelArguments) {
    if (const auto *Buffer = std::get_if<const DeviceBuffer *>(&Argument)) {
      if (!*Buffer)
        return Error{ErrorKind::InvalidArgument,
                     "kernel argument has a null buffer"};
      if (auto Err = validateBuffer(**Buffer))
        return Err;
    } else if (!std::get<const void *>(Argument)) {
      return Error{ErrorKind::InvalidArgument, "kernel scalar storage is null"};
    }
  }

  return Error::success();
}

} // namespace gpu_libm_bench::cuda
