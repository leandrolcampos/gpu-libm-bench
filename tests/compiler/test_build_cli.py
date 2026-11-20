# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Test build selection, configuration overrides, and dispatch."""

import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from hydra.core.config_store import ConfigStore
from hydra.errors import ConfigCompositionException

from gpu_libm_bench.compiler import __main__ as build_cli
from gpu_libm_bench.compiler._cmake import CMakeBuild
from gpu_libm_bench.compiler.backends import BackendModuleNotFoundError
from gpu_libm_bench.compiler.libraries import LibraryUnavailableError, ResolvedLibrary
from gpu_libm_bench.compiler.targets import CompilationTarget, TargetUnavailableError
from gpu_libm_bench.compiler.toolchains import (
    BackendSDKUnavailableError,
    CUDAToolkit,
    LLVMToolchain,
    ROCmCoreSDK,
)
from gpu_libm_bench.configs import BackendConfig, LibraryBitcodeConfig, LibraryConfig
from gpu_libm_bench.devices import DeviceInfo, TimerInfo


@pytest.fixture
def compiler(llvm: LLVMToolchain, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend_sdk = CUDAToolkit(
        tmp_path / "backend_sdk", tmp_path / "backend_sdk/bin/ptxas", "13.4"
    )
    monkeypatch.setattr(
        build_cli,
        "resolve_backend_sdk",
        Mock(
            return_value=backend_sdk,
            side_effect=lambda **kwargs: (
                backend_sdk
                if kwargs["backend"] == "cuda"
                else ROCmCoreSDK(tmp_path / "rocm", "test")
            ),
        ),
    )
    monkeypatch.setattr(build_cli, "resolve_llvm", Mock(return_value=llvm))

    monkeypatch.setattr(build_cli, "build_backend", Mock())
    monkeypatch.setattr(build_cli, "configure_backend", Mock())

    monkeypatch.setattr(build_cli.DeviceContext, "device_count", Mock(return_value=1))
    monkeypatch.setattr(
        build_cli.DeviceContext,
        "get_device_info",
        Mock(
            return_value=DeviceInfo(
                "cuda",
                3,
                "GPU",
                "sm_90",
                16,
                2_000_000_000,
                1024,
                TimerInfo("timer", 64, 10**9),
            )
        ),
    )
    monkeypatch.setattr(
        build_cli,
        "resolve_target",
        Mock(
            side_effect=lambda **kwargs: CompilationTarget(
                kwargs["backend"],
                kwargs["target_triple"],
                kwargs["architecture"] or kwargs["device_info"].architecture,
            )
        ),
    )

    def resolve_library(*, library_config, llvm, target, backend_sdk):
        bitcode_path = library_config.backends[target.backend].bitcode_path
        path = (
            Path(bitcode_path)
            if bitcode_path is not None
            else tmp_path / f"{library_config.name}.bc"
        )
        return ResolvedLibrary(library_config.name, path)

    monkeypatch.setattr(build_cli, "resolve_library", Mock(side_effect=resolve_library))
    monkeypatch.setattr(build_cli, "build_kernels", Mock())
    monkeypatch.setattr(
        build_cli,
        "configure_kernels",
        Mock(
            side_effect=lambda **kwargs: CMakeBuild(
                tmp_path / "kernels", kwargs["build_directory"]
            )
        ),
    )
    monkeypatch.setattr(build_cli, "run_tool", Mock())

    def run(project="kernels", *arguments):
        monkeypatch.setattr(sys, "argv", ["compiler", project, *arguments])
        build_cli.main()

    return run


@pytest.fixture
def catalogue(monkeypatch: pytest.MonkeyPatch):
    store = ConfigStore.instance()
    monkeypatch.setattr(store, "repo", deepcopy(store.repo))
    store.store(
        group="backend",
        name="_test_other",
        package="backend",
        node=BackendConfig("hip", "amdgcn-amd-amdhsa", "gfx1100", 0, None),
        replace=True,
    )
    store.store(
        group="library",
        name="variant",
        node=LibraryConfig(
            "variant",
            {backend: LibraryBitcodeConfig(None) for backend in ("cuda", "hip")},
        ),
        replace=True,
    )


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ("library=[llvm-libm]", "library must be a single name"),
        ("backend=[cuda]", "backend must be a single name"),
        ("library=missing", "unknown library"),
    ],
)
def test_invalid_selection_is_rejected(compiler, selection: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compiler("kernels", selection)

    build_cli.resolve_backend_sdk.assert_not_called()


@pytest.mark.parametrize(
    "override", ["function=expf", "metric=latency", "run.seed=123"]
)
def test_experiment_settings_are_rejected(compiler, override: str) -> None:
    with pytest.raises(ConfigCompositionException):
        compiler("kernels", "backend=cuda", override)

    build_cli.resolve_backend_sdk.assert_not_called()


def test_backend_command_requires_only_backend_config(compiler, tmp_path: Path) -> None:
    compiler(
        "backend",
        "--configure-only",
        "--build-tests",
        "--build-root",
        str(tmp_path / "custom"),
        "backend=cuda",
        f"backend.sdk_root={tmp_path / 'backend_sdk'}",
    )

    build_cli.resolve_backend_sdk.assert_called_once_with(
        backend="cuda", root=tmp_path / "backend_sdk"
    )
    build_cli.configure_backend.assert_called_once_with(
        "cuda",
        backend_sdk=build_cli.resolve_backend_sdk.return_value,
        build_directory=tmp_path / "custom" / "cuda",
        build_tests=True,
    )
    build_cli.resolve_llvm.assert_not_called()


@pytest.mark.parametrize("explicit", [False, True])
def test_no_available_backends_is_an_error(compiler, explicit: bool) -> None:
    build_cli.resolve_backend_sdk.side_effect = BackendSDKUnavailableError(
        "backend SDK is missing"
    )
    with pytest.raises(
        RuntimeError,
        match="backend SDK is missing" if explicit else "no native backends",
    ):
        compiler("backend", *(["backend=cuda"] if explicit else []))


@pytest.mark.parametrize("project", ["backend", "kernels"])
def test_automatic_backend_discovery_skips_only_missing_sdks(
    compiler, catalogue, project: str
) -> None:
    backend_sdk = build_cli.resolve_backend_sdk.return_value

    def resolve(*, backend, root):
        if backend != "cuda":
            raise BackendSDKUnavailableError("backend SDK is missing")
        return backend_sdk

    build_cli.resolve_backend_sdk.side_effect = resolve

    compiler(project)

    calls = (
        build_cli.build_backend.call_args_list
        if project == "backend"
        else build_cli.build_kernels.call_args_list
    )
    assert calls
    assert all(
        (call.args[0] if project == "backend" else call.kwargs["target"].backend)
        == "cuda"
        for call in calls
    )


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (
            [],
            {
                ("cuda", "libdevice"),
                ("cuda", "llvm-libm"),
                ("cuda", "variant"),
                ("hip", "variant"),
            },
        ),
        (
            ["backend=cuda"],
            {("cuda", "libdevice"), ("cuda", "llvm-libm"), ("cuda", "variant")},
        ),
        (["library=variant"], {("cuda", "variant"), ("hip", "variant")}),
        (["backend=cuda", "library=variant"], {("cuda", "variant")}),
    ],
)
def test_kernel_discovery_and_filters(
    compiler, catalogue, tmp_path: Path, filters, expected
) -> None:
    compiler(
        "kernels",
        "--build-root",
        str(tmp_path),
        "backend.architecture=sm_90",
        *filters,
    )

    calls = build_cli.build_kernels.call_args_list
    expected = expected | {(backend, "support") for backend, _ in expected}
    assert {
        (call.kwargs["target"].backend, call.kwargs["resolved_library"].name)
        for call in calls
    } == expected
    assert len(calls) == len(expected)
    assert len({call.kwargs["build_directory"] for call in calls}) == len(calls)


@pytest.mark.parametrize("architecture", [None, "sm_120"])
def test_kernel_command_preserves_build_selection(
    compiler,
    llvm: LLVMToolchain,
    tmp_path: Path,
    architecture: str | None,
) -> None:
    root = tmp_path / "custom"
    bitcode_path = tmp_path / "custom.bc"
    compiler(
        "kernels",
        "--build-root",
        str(root),
        "--build-tests",
        "backend=cuda",
        "library=llvm-libm",
        "backend.device_id=3",
        f"backend.architecture={architecture or 'null'}",
        f"llvm.root={llvm.root}",
        f"library.backends.cuda.bitcode_path={bitcode_path}",
        "library.functions.expf.symbol=my_expf",
    )

    calls = build_cli.build_kernels.call_args_list
    support, call = (item.kwargs for item in calls)
    target = call["target"]
    library_config = build_cli.resolve_library.call_args.kwargs["library_config"]

    assert target.architecture == (architecture or "sm_90")
    assert library_config.symbol_for("expf") == "my_expf"
    assert call["resolved_library"] == ResolvedLibrary("llvm-libm", bitcode_path)
    assert call["build_directory"] == root / "cuda" / target.architecture / "llvm-libm"
    assert call["build_tests"] is True
    assert support["resolved_library"] == ResolvedLibrary("support")
    assert support["target"] == target
    assert support["build_directory"] == root / "cuda" / target.architecture / "support"
    build_cli.resolve_llvm.assert_called_once_with(llvm.root)

    if architecture is None:
        build_cli.DeviceContext.get_device_info.assert_called_once_with(
            3, backend="cuda"
        )
    else:
        build_cli.DeviceContext.device_count.assert_not_called()
        build_cli.DeviceContext.get_device_info.assert_not_called()


@pytest.mark.parametrize("mode", ["--configure-only", "--tidy"])
def test_configure_and_tidy_use_each_selected_build_directory(
    compiler, catalogue, tmp_path: Path, mode: str
) -> None:
    compiler(
        "kernels",
        "backend=cuda",
        "--build-root",
        str(tmp_path),
        mode,
        "--build-tests",
    )

    build_cli.build_kernels.assert_not_called()
    builds = build_cli.configure_kernels.call_args_list
    assert [call.kwargs["resolved_library"].name for call in builds] == [
        "support",
        "libdevice",
        "llvm-libm",
        "variant",
    ]
    assert all(call.kwargs["build_tests"] for call in builds)

    if mode == "--tidy":
        assert [call.args[0][-1] for call in build_cli.run_tool.call_args_list] == [
            call.kwargs["build_directory"] for call in builds
        ]
    else:
        build_cli.run_tool.assert_not_called()


def test_no_device_requires_explicit_architecture(compiler) -> None:
    build_cli.DeviceContext.device_count.return_value = 0

    with pytest.raises(TargetUnavailableError, match="set backend.architecture"):
        compiler("kernels", "backend=cuda")


def test_architecture_discovery_builds_only_a_missing_backend(compiler) -> None:
    build_cli.DeviceContext.device_count.side_effect = [
        BackendModuleNotFoundError("missing"),
        1,
    ]

    compiler("kernels", "backend=cuda")

    build_cli.build_backend.assert_called_once_with(
        "cuda",
        backend_sdk=build_cli.resolve_backend_sdk.return_value,
    )


@pytest.mark.parametrize("explicit_backend", [False, True])
def test_library_backend_support_is_respected(
    compiler, catalogue, explicit_backend: bool
) -> None:
    arguments = [
        "~library.backends.cuda",
        "library=variant",
    ]
    if explicit_backend:
        with pytest.raises(
            LibraryUnavailableError, match="does not support backend 'cuda'"
        ):
            compiler("kernels", *arguments, "backend=cuda")
    else:
        compiler("kernels", *arguments)
        assert {
            call.kwargs["target"].backend
            for call in build_cli.build_kernels.call_args_list
        } == {"hip"}


@pytest.mark.parametrize("explicit", [False, True])
def test_unavailable_target_is_skipped_only_for_automatic_backends(
    compiler, catalogue, explicit: bool
) -> None:
    resolve = build_cli.resolve_target.side_effect

    def available(**kwargs):
        if kwargs["backend"] == "cuda":
            raise TargetUnavailableError("Clang does not support this architecture")
        return resolve(**kwargs)

    build_cli.resolve_target.side_effect = available

    if explicit:
        with pytest.raises(TargetUnavailableError, match="Clang does not support"):
            compiler("kernels", "backend=cuda")
    else:
        compiler("kernels")
        assert {
            call.kwargs["target"].backend
            for call in build_cli.build_kernels.call_args_list
        } == {"hip"}


@pytest.mark.parametrize("explicit", [False, True])
def test_unavailable_library_is_skipped_only_when_selection_can_be_satisfied(
    compiler,
    catalogue,
    explicit: bool,
) -> None:
    resolve = build_cli.resolve_library.side_effect

    def unavailable(**kwargs):
        if kwargs["library_config"].name == "variant":
            raise LibraryUnavailableError("bitcode is missing")
        return resolve(**kwargs)

    build_cli.resolve_library.side_effect = unavailable

    if explicit:
        with pytest.raises(LibraryUnavailableError, match="bitcode is missing"):
            compiler("kernels", "backend=cuda", "library=variant")
    else:
        compiler("kernels", "backend=cuda")

        assert [
            call.kwargs["resolved_library"].name
            for call in build_cli.build_kernels.call_args_list
        ] == ["support", "libdevice", "llvm-libm"]


def test_explicit_library_must_be_available_on_at_least_one_backend(
    compiler, catalogue
) -> None:
    build_cli.resolve_library.side_effect = LibraryUnavailableError(
        "bitcode is missing"
    )
    with pytest.raises(
        LibraryUnavailableError,
        match="selected math library is unavailable: variant",
    ):
        compiler(
            "kernels",
            "backend.architecture=sm_90",
            "library=variant",
        )


def test_no_available_kernel_combinations_is_an_error(compiler) -> None:
    build_cli.resolve_library.side_effect = LibraryUnavailableError(
        "bitcode is missing"
    )
    with pytest.raises(
        LibraryUnavailableError, match="no library/backend combinations"
    ):
        compiler()


@pytest.mark.parametrize(
    "operation", ["resolve_backend_sdk", "resolve_library", "build_kernels"]
)
def test_real_discovery_and_build_failures_are_propagated(
    compiler, operation: str
) -> None:
    error = RuntimeError(f"{operation} failed")
    getattr(build_cli, operation).side_effect = error

    with pytest.raises(RuntimeError) as caught:
        compiler()

    assert caught.value is error
