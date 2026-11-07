# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for experiment library eligibility and selection."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_module

from gpu_libm_bench.compiler._cmake import CMakeBuild
from gpu_libm_bench.compiler.libraries import (
    LibraryBitcode,
    LibraryUnavailableError,
    resolve_library,
)
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import CUDAToolkit, LLVMToolchain
from gpu_libm_bench.configs import (
    ExperimentConfig,
    LibraryConfig,
    register_config_schema,
    validate_experiment_config,
)
from gpu_libm_bench.experiment import _libraries


@pytest.fixture
def config() -> ExperimentConfig:
    register_config_schema()
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        config = validate_experiment_config(
            compose(
                config_name="experiment_config",
                overrides=["function=expf", "metric=latency", "backend=cuda"],
            )
        )

    backends = config.libraries["llvm-libm"].backends
    return replace(
        config,
        libraries={
            name: LibraryConfig(
                name,
                {backend: replace(bitcode) for backend, bitcode in backends.items()},
            )
            for name in ("first", "second")
        },
    )


@pytest.fixture
def select(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    llvm = LLVMToolchain(
        tmp_path,
        tmp_path / "clang++",
        tmp_path / "llvm-dis",
        tmp_path / "llvm-nm",
        (tmp_path / "lib",),
        "test LLVM",
    )
    sdk = CUDAToolkit(tmp_path / "cuda", tmp_path / "cuda/bin/ptxas", "13.4")
    target = CompilationTarget("cuda", "nvptx64-nvidia-cuda", "sm_120")

    def resolve(*, library, bitcode_path, **kwargs):
        return LibraryBitcode(library, bitcode_path or tmp_path / f"{library}.bc")

    def configure(*, bitcode, target, build_directory, **kwargs):
        plan = build_directory / "build-plan.json"
        plan.parent.mkdir(parents=True)

        plan.write_text(
            json.dumps(
                {
                    "library": bitcode.library,
                    "backend": target.backend,
                    "architecture": target.architecture,
                    "target_triple": target.target_triple,
                    "functions": ["expf"],
                }
            )
        )

        return CMakeBuild(tmp_path / "kernels", build_directory)

    monkeypatch.setattr(_libraries, "resolve_library", resolve)
    monkeypatch.setattr(_libraries, "configure_kernels", configure)

    def select(config):
        return _libraries.select_libraries(
            config,
            llvm=llvm,
            target=target,
            sdk=sdk,
            build_root=tmp_path / "builds",
        )

    return select


def test_invalid_function_signature_is_rejected_before_preparing_libraries(
    config: ExperimentConfig,
    select,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.function.signature.return_type = "double"
    resolve = Mock()
    monkeypatch.setattr(_libraries, "resolve_library", resolve)

    with pytest.raises(ValueError, match="return type must match"):
        select(config)

    resolve.assert_not_called()


@pytest.mark.parametrize("names", [None, ["second", "first"]])
def test_library_selection_preserves_order_and_bitcode_overrides(
    config: ExperimentConfig,
    select,
    names: list[str] | None,
    tmp_path: Path,
) -> None:
    config.run.libraries = names
    config.libraries["first"].backends["cuda"].bitcode_path = "/custom/first.bc"

    selected = select(config)

    assert [bitcode.library for bitcode in selected] == (names or ["first", "second"])

    paths = {bitcode.library: bitcode.bitcode_path for bitcode in selected}
    assert paths == {
        "first": Path("/custom/first.bc"),
        "second": tmp_path / "second.bc",
    }


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("missing", ["backend", "bitcode", "wrappers"])
def test_unavailable_library_is_skipped_only_for_automatic_selection(
    config: ExperimentConfig,
    select,
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
    missing: str,
) -> None:
    config.run.libraries = ["first", "second"] if explicit else None

    if missing == "backend":
        config.libraries["first"].backends = {}
    elif missing == "bitcode":
        resolve = _libraries.resolve_library

        def unavailable(*, library, **kwargs):
            if library == "first":
                raise LibraryUnavailableError("bitcode is missing")
            return resolve(library=library, **kwargs)

        monkeypatch.setattr(_libraries, "resolve_library", unavailable)
    else:
        read = _libraries.read_registered_functions
        monkeypatch.setattr(
            _libraries,
            "read_registered_functions",
            lambda build, *, library, target: (
                ()
                if library == "first"
                else read(build, library=library, target=target)
            ),
        )

    if explicit:
        with pytest.raises(
            LibraryUnavailableError, match="library 'first' is unavailable"
        ):
            select(config)
    else:
        assert [bitcode.library for bitcode in select(config)] == ["second"]


def test_no_eligible_libraries_is_an_error(config: ExperimentConfig, select) -> None:
    for library in config.libraries.values():
        library.backends = {}

    with pytest.raises(
        LibraryUnavailableError, match="no math libraries are available"
    ):
        select(config)


def test_unsupported_library_is_not_skipped_in_automatic_selection(
    config: ExperimentConfig, select, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_libraries, "resolve_library", resolve_library)

    with pytest.raises(ValueError, match="unsupported math library: 'first'"):
        select(config)


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        ("resolve_library", RuntimeError("bitcode tool failed")),
        ("configure_kernels", RuntimeError("CMake failed")),
        ("read_registered_functions", ValueError("invalid build plan")),
    ],
)
def test_unexpected_failures_are_propagated(
    config: ExperimentConfig,
    select,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    error: Exception,
) -> None:
    monkeypatch.setattr(_libraries, operation, Mock(side_effect=error))

    with pytest.raises(type(error)) as caught:
        select(config)

    assert caught.value is error
