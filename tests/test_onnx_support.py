import sys
import warnings
import types
from pathlib import Path

import numpy as np

import main
import utils.onnx_runtime as onnx_runtime_mod
from utils.onnx_runtime import build_provider_candidates, build_session
from utils.detector import deduplicate_detections


def test_prefer_onnx_sibling_uses_exported_model(tmp_path):
    pt_path = tmp_path / "yolo11x.pt"
    onnx_path = tmp_path / "yolo11x.onnx"
    pt_path.write_text("pt")
    onnx_path.write_text("onnx")

    assert main._prefer_onnx_sibling(str(pt_path)) == str(onnx_path)


def test_build_detector_uses_onnx_backend_without_model_to(monkeypatch, tmp_path):
    class DummyYOLO:
        def __init__(self, model_path):
            self.model_path = model_path
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=DummyYOLO))

    onnx_path = tmp_path / "detector.onnx"
    onnx_path.write_text("onnx")
    model, backend = main._build_detector({"model": str(onnx_path)}, "cuda:0")

    assert backend == "onnx"
    assert model.model_path == str(onnx_path)
    assert model.to_calls == []


def test_build_detector_prefers_existing_onnx_sibling(monkeypatch, tmp_path):
    class DummyYOLO:
        def __init__(self, model_path):
            self.model_path = model_path
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=DummyYOLO))

    pt_path = tmp_path / "detector.pt"
    onnx_path = tmp_path / "detector.onnx"
    pt_path.write_text("pt")
    onnx_path.write_text("onnx")

    model, backend = main._build_detector({"model": str(pt_path)}, "cuda:0")

    assert backend == "onnx"
    assert model.model_path == str(onnx_path)
    assert model.to_calls == []


def test_build_reid_embedder_uses_onnx_when_sibling_exists(monkeypatch, tmp_path):
    import reid.embedder as embedder_mod

    class DummyONNX:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyPT:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(embedder_mod, "OSNetONNXEmbedder", DummyONNX)
    monkeypatch.setattr(embedder_mod, "OSNetEmbedder", DummyPT)

    pth_path = tmp_path / "osnet_x1_0_msmt17.pth"
    onnx_path = tmp_path / "osnet_x1_0_msmt17.onnx"
    pth_path.write_text("pth")
    onnx_path.write_text("onnx")

    compute_cfg = {"onnx_runtime": {"provider": "cpu"}}
    embedder = main._build_reid_embedder(
        {
            "type": "osnet",
            "model": "osnet_x1_0",
            "osnet_weights": str(pth_path),
            "tta": False,
        },
        device="cpu",
        half=False,
        compute_cfg=compute_cfg,
    )

    assert isinstance(embedder, DummyONNX)
    assert embedder.kwargs["weights_path"] == str(onnx_path)
    assert embedder.kwargs["compute_cfg"] == compute_cfg


def test_build_reid_embedder_keeps_pth_backend_without_onnx(monkeypatch, tmp_path):
    import reid.embedder as embedder_mod

    class DummyONNX:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyPT:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(embedder_mod, "OSNetONNXEmbedder", DummyONNX)
    monkeypatch.setattr(embedder_mod, "OSNetEmbedder", DummyPT)

    pth_path = tmp_path / "osnet_x1_0_msmt17.pth"
    pth_path.write_text("pth")

    embedder = main._build_reid_embedder(
        {
            "type": "osnet",
            "model": "osnet_x1_0",
            "osnet_weights": str(pth_path),
            "tta": False,
        },
        device="cpu",
        half=False,
    )

    assert isinstance(embedder, DummyPT)
    assert embedder.kwargs["weights_path"] == str(pth_path)


def test_build_detector_uses_rfdetr_backend_when_onnx_matches(monkeypatch, tmp_path):
    class DummyRFDETR:
        def __init__(self, weights_path, device="cpu", half=False, compute_cfg=None):
            self.weights_path = weights_path
            self.device = device
            self.half = half
            self.compute_cfg = compute_cfg
            self.provider_mode = "cuda"
            self.providers = ("CPUExecutionProvider",)

    monkeypatch.setattr(main, "is_rfdetr_onnx_model", lambda path: True)
    monkeypatch.setattr(main, "RFDETRONNXDetector", DummyRFDETR)

    onnx_path = tmp_path / "checkpoint_best_ema.onnx"
    onnx_path.write_text("onnx")

    compute_cfg = {"onnx_runtime": {"provider": "tensorrt"}}
    model, backend = main._build_detector(
        {"model": str(onnx_path)},
        "cuda:0",
        half=True,
        compute_cfg=compute_cfg,
    )

    assert backend == "rfdetr-onnx"
    assert isinstance(model, DummyRFDETR)
    assert model.weights_path == str(onnx_path)
    assert model.device == "cuda:0"
    assert model.half is True
    assert model.compute_cfg == compute_cfg


def test_run_detector_uses_rfdetr_predict():
    class DummyRFDETR:
        def __init__(self):
            self.calls = []

        def predict(self, frame, conf, classes, dedup_cfg=None):
            self.calls.append((frame.shape, conf, classes, dedup_cfg))
            return np.array([[1, 2, 3, 4, 0.9, 0]], dtype=np.float32)

    model = DummyRFDETR()
    frame = np.zeros((16, 20, 3), dtype=np.uint8)

    dets = main._run_detector(
        model,
        "rfdetr-onnx",
        frame,
        {"confidence": 0.25, "classes": [0], "dedup": {"enabled": True, "iou": 0.7}},
        device="cpu",
        half=False,
    )

    assert dets.shape == (1, 6)
    assert model.calls == [((16, 20, 3), 0.25, [0], {"enabled": True, "iou": 0.7})]


class _DummySession:
    def __init__(self, providers):
        self._providers = tuple(providers)

    def get_providers(self):
        return list(self._providers)


class _DummySessionOptions:
    def __init__(self):
        self.graph_optimization_level = None


class _DummyGraphOpt:
    ORT_ENABLE_ALL = "all"


class _DummyRuntime:
    SessionOptions = _DummySessionOptions
    GraphOptimizationLevel = _DummyGraphOpt

    def __init__(self, available_providers, provider_results):
        self._available_providers = list(available_providers)
        self._provider_results = provider_results
        self.calls = []

    def get_available_providers(self):
        return list(self._available_providers)

    def InferenceSession(self, model_path, sess_options=None, providers=None):
        names = tuple(provider[0] if isinstance(provider, tuple) else provider for provider in providers)
        self.calls.append((model_path, names, sess_options.graph_optimization_level if sess_options else None))
        result = self._provider_results[names]
        if isinstance(result, Exception):
            raise result
        return _DummySession(result)


def test_build_provider_candidates_prefers_tensorrt_then_cuda(tmp_path):
    runtime = _DummyRuntime(
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        {},
    )

    chains, mode = build_provider_candidates(
        runtime,
        device="cuda:1",
        half=True,
        compute_cfg={
            "onnx_runtime": {
                "provider": "tensorrt",
                "trt_engine_cache_path": str(tmp_path / "trt_cache"),
            }
        },
    )

    assert mode == "tensorrt"
    assert [provider[0] if isinstance(provider, tuple) else provider for provider in chains[0]][:2] == [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
    ]
    trt_name, trt_options = chains[0][0]
    assert trt_name == "TensorrtExecutionProvider"
    assert trt_options["device_id"] == 1
    assert trt_options["trt_fp16_enable"] is True
    assert Path(trt_options["trt_engine_cache_path"]).exists()


def test_build_session_falls_back_from_tensorrt_to_cuda():
    runtime = _DummyRuntime(
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        {
            ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"): ["CPUExecutionProvider"],
            ("CUDAExecutionProvider", "CPUExecutionProvider"): ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ("CPUExecutionProvider",): ["CPUExecutionProvider"],
        },
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session, active_providers, mode = build_session(
            runtime,
            "model.onnx",
            device="cuda:0",
            half=True,
            compute_cfg={"onnx_runtime": {"provider": "tensorrt"}},
            model_label="RF-DETR ONNX",
        )

    assert isinstance(session, _DummySession)
    assert mode == "tensorrt"
    assert active_providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert runtime.calls[0][1] == ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
    assert runtime.calls[1][1] == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert runtime.calls[1][2] == _DummyGraphOpt.ORT_ENABLE_ALL
    assert any("Using providers=['CUDAExecutionProvider', 'CPUExecutionProvider']" in str(w.message) for w in caught)



def test_build_provider_candidates_uses_cpu_only_when_device_is_cpu():
    runtime = _DummyRuntime(
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        {},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chains, mode = build_provider_candidates(
            runtime,
            device="cpu",
            half=True,
            compute_cfg={"onnx_runtime": {"provider": "tensorrt"}},
        )

    assert mode == "tensorrt"
    assert chains == [["CPUExecutionProvider"]]
    assert any("Using CPUExecutionProvider" in str(w.message) for w in caught)


def test_build_session_falls_back_all_the_way_to_cpu_after_gpu_failures():
    runtime = _DummyRuntime(
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        {
            ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"): RuntimeError("trt fail"),
            ("CUDAExecutionProvider", "CPUExecutionProvider"): RuntimeError("cuda fail"),
            ("CPUExecutionProvider",): ["CPUExecutionProvider"],
        },
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session, active_providers, mode = build_session(
            runtime,
            "model.onnx",
            device="cuda:0",
            half=True,
            compute_cfg={"onnx_runtime": {"provider": "tensorrt"}},
            model_label="RF-DETR ONNX",
        )

    assert isinstance(session, _DummySession)
    assert mode == "tensorrt"
    assert active_providers == ("CPUExecutionProvider",)
    assert runtime.calls[-1][1] == ("CPUExecutionProvider",)
    assert any("Using providers=['CPUExecutionProvider']" in str(w.message) for w in caught)


def test_preload_runtime_libraries_loads_discovered_dirs_once(tmp_path, monkeypatch):
    site_dir = tmp_path / "site"
    nvidia_lib = site_dir / "nvidia" / "cublas" / "lib"
    trt_lib = site_dir / "tensorrt_libs"
    nvidia_lib.mkdir(parents=True)
    trt_lib.mkdir(parents=True)
    (nvidia_lib / "libcublas.so.12").write_text("")
    (trt_lib / "libnvinfer.so.10").write_text("")

    calls = []
    monkeypatch.setattr(onnx_runtime_mod, "_PRELOAD_ATTEMPTED", False)
    monkeypatch.setattr(onnx_runtime_mod.sys, "path", [str(site_dir)])
    monkeypatch.setattr(
        onnx_runtime_mod.ctypes,
        "CDLL",
        lambda path, mode=None: calls.append((Path(path).name, mode)),
    )

    onnx_runtime_mod._preload_runtime_libraries("cuda:0")
    onnx_runtime_mod._preload_runtime_libraries("cuda:0")

    assert [name for name, _ in calls] == ["libcublas.so.12", "libnvinfer.so.10"]


def test_deduplicate_detections_suppresses_near_identical_person_boxes():
    dets = np.array([
        [568.0, 256.2, 640.0, 445.6, 0.80, 0.0],
        [563.8, 305.9, 640.0, 444.8, 0.79, 0.0],
    ], dtype=np.float32)

    filtered = deduplicate_detections(dets)

    assert filtered.shape == (1, 6)
    np.testing.assert_allclose(filtered[0], dets[0])


def test_deduplicate_detections_keeps_close_distinct_people():
    dets = np.array([
        [100.0, 120.0, 190.0, 280.0, 0.90, 0.0],
        [145.0, 120.0, 235.0, 280.0, 0.88, 0.0],
    ], dtype=np.float32)

    filtered = deduplicate_detections(dets)

    assert filtered.shape == (2, 6)
