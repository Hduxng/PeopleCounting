"""Shared ONNX Runtime provider selection helpers with TensorRT fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import ctypes
import sys
import warnings


ProviderSpec = Any
_PRELOAD_ATTEMPTED = False


def _provider_name(provider: ProviderSpec) -> str:
    if isinstance(provider, tuple):
        return str(provider[0])
    return str(provider)


def _provider_chain_names(providers: list[ProviderSpec]) -> tuple[str, ...]:
    return tuple(_provider_name(provider) for provider in providers)


def _cuda_device_id(device: str) -> int:
    if not isinstance(device, str) or not device.startswith("cuda"):
        return 0
    if ":" not in device:
        return 0
    try:
        return max(0, int(device.split(":", 1)[1]))
    except (TypeError, ValueError):
        return 0


def _runtime_cfg(compute_cfg: dict | None, half: bool) -> dict[str, Any]:
    runtime_cfg = dict((compute_cfg or {}).get("onnx_runtime", {}) or {})
    provider = str(runtime_cfg.get("provider", "auto")).strip().lower()
    if provider not in {"auto", "tensorrt", "cuda", "cpu"}:
        warnings.warn(
            f"Unknown compute.onnx_runtime.provider={provider!r}. Falling back to 'auto'.",
            RuntimeWarning,
        )
        provider = "auto"

    cache_path = runtime_cfg.get("trt_engine_cache_path", "output/trt_cache")
    timing_cache_path = runtime_cfg.get("trt_timing_cache_path", cache_path)
    return {
        "provider": provider,
        "trt_fp16": bool(runtime_cfg.get("trt_fp16", half)),
        "trt_engine_cache": bool(runtime_cfg.get("trt_engine_cache", True)),
        "trt_engine_cache_path": str(cache_path),
        "trt_timing_cache": bool(runtime_cfg.get("trt_timing_cache", True)),
        "trt_timing_cache_path": str(timing_cache_path),
    }


def _build_tensorrt_provider(device_id: int, runtime_cfg: dict[str, Any]) -> ProviderSpec:
    options: dict[str, Any] = {"device_id": device_id}
    if runtime_cfg["trt_fp16"]:
        options["trt_fp16_enable"] = True

    if runtime_cfg["trt_engine_cache"]:
        cache_dir = Path(runtime_cfg["trt_engine_cache_path"]).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        options["trt_engine_cache_enable"] = True
        options["trt_engine_cache_path"] = str(cache_dir)

    if runtime_cfg["trt_timing_cache"]:
        timing_cache_dir = Path(runtime_cfg["trt_timing_cache_path"]).expanduser().resolve()
        timing_cache_dir.mkdir(parents=True, exist_ok=True)
        options["trt_timing_cache_enable"] = True
        options["trt_timing_cache_path"] = str(timing_cache_dir)

    return ("TensorrtExecutionProvider", options)


def _candidate_library_dirs() -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for entry in sys.path:
        if not entry:
            continue
        base = Path(entry)
        if not base.exists() or not base.is_dir():
            continue

        nvidia_root = base / "nvidia"
        if nvidia_root.exists():
            for lib_dir in sorted(nvidia_root.glob("*/lib")):
                resolved = lib_dir.resolve()
                if resolved not in seen:
                    roots.append(resolved)
                    seen.add(resolved)

        tensorrt_root = base / "tensorrt_libs"
        if tensorrt_root.exists() and tensorrt_root.is_dir():
            resolved = tensorrt_root.resolve()
            if resolved not in seen:
                roots.append(resolved)
                seen.add(resolved)
    return roots


def _preload_runtime_libraries(device: str) -> None:
    global _PRELOAD_ATTEMPTED
    if _PRELOAD_ATTEMPTED or not device.startswith("cuda"):
        return

    _PRELOAD_ATTEMPTED = True
    for lib_dir in _candidate_library_dirs():
        for path in sorted(lib_dir.glob("*.so*")):
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


def build_provider_candidates(
    runtime,
    device: str,
    half: bool = False,
    compute_cfg: dict | None = None,
) -> tuple[list[list[ProviderSpec]], str]:
    available = set(runtime.get_available_providers())
    runtime_cfg = _runtime_cfg(compute_cfg, half)
    mode = runtime_cfg["provider"]
    cpu_chain: list[ProviderSpec] = ["CPUExecutionProvider"]

    if not device.startswith("cuda"):
        if mode != "cpu":
            warnings.warn(
                f"ONNX Runtime provider {mode!r} requested while compute.device={device!r}. Using CPUExecutionProvider.",
                RuntimeWarning,
            )
        return [cpu_chain], mode

    device_id = _cuda_device_id(device)
    cuda_provider: ProviderSpec | None = None
    if "CUDAExecutionProvider" in available:
        cuda_provider = ("CUDAExecutionProvider", {"device_id": device_id})

    trt_provider: ProviderSpec | None = None
    if "TensorrtExecutionProvider" in available:
        trt_provider = _build_tensorrt_provider(device_id, runtime_cfg)

    chains: list[list[ProviderSpec]] = []
    if mode in {"auto", "tensorrt"}:
        if trt_provider is not None:
            chain = [trt_provider]
            if cuda_provider is not None:
                chain.append(cuda_provider)
            chain.append("CPUExecutionProvider")
            chains.append(chain)
        elif mode == "tensorrt":
            warnings.warn(
                "TensorRT requested for ONNX Runtime, but TensorrtExecutionProvider is unavailable. Falling back.",
                RuntimeWarning,
            )

    if mode in {"auto", "tensorrt", "cuda"}:
        if cuda_provider is not None:
            chains.append([cuda_provider, "CPUExecutionProvider"])
        elif mode == "cuda":
            warnings.warn(
                "CUDA requested for ONNX Runtime, but CUDAExecutionProvider is unavailable. Falling back to CPUExecutionProvider.",
                RuntimeWarning,
            )

    if not chains or _provider_chain_names(chains[-1]) != ("CPUExecutionProvider",):
        chains.append(cpu_chain)

    deduped: list[list[ProviderSpec]] = []
    seen: set[tuple[str, ...]] = set()
    for chain in chains:
        names = _provider_chain_names(chain)
        if names in seen:
            continue
        deduped.append(chain)
        seen.add(names)
    return deduped, mode


def _session_matches_requested_provider(
    providers: list[ProviderSpec],
    active_providers: tuple[str, ...],
    device: str,
) -> bool:
    if not device.startswith("cuda"):
        return True

    requested = _provider_chain_names(providers)
    if "TensorrtExecutionProvider" in requested:
        return "TensorrtExecutionProvider" in active_providers
    if "CUDAExecutionProvider" in requested:
        return any(provider in active_providers for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider"))
    return True


def build_session(
    runtime,
    model_path: str | Path,
    device: str,
    half: bool = False,
    compute_cfg: dict | None = None,
    model_label: str = "ONNX model",
):
    _preload_runtime_libraries(device)
    provider_candidates, mode = build_provider_candidates(
        runtime,
        device=device,
        half=half,
        compute_cfg=compute_cfg,
    )

    sess_options = runtime.SessionOptions() if hasattr(runtime, "SessionOptions") else None
    graph_opt = getattr(getattr(runtime, "GraphOptimizationLevel", None), "ORT_ENABLE_ALL", None)
    if sess_options is not None and graph_opt is not None:
        sess_options.graph_optimization_level = graph_opt

    fallback_notes: list[str] = []
    for providers in provider_candidates:
        kwargs = {"providers": providers}
        if sess_options is not None:
            kwargs["sess_options"] = sess_options

        try:
            session = runtime.InferenceSession(str(model_path), **kwargs)
        except Exception as exc:
            fallback_notes.append(f"{list(_provider_chain_names(providers))} init failed: {exc}")
            continue

        active_providers = tuple(session.get_providers())
        if not _session_matches_requested_provider(providers, active_providers, device):
            fallback_notes.append(
                f"{list(_provider_chain_names(providers))} activated {list(active_providers)}"
            )
            continue

        if fallback_notes:
            warnings.warn(
                f"{model_label}: {'; '.join(fallback_notes)}. Using providers={list(active_providers)}.",
                RuntimeWarning,
            )
        return session, active_providers, mode

    failure_text = "; ".join(fallback_notes) or "no provider chain succeeded"
    raise RuntimeError(f"{model_label}: unable to create ONNX Runtime session. {failure_text}")
