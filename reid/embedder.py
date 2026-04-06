"""
Re-ID feature extractors for person tracking.

Available embedders:
  - CLIPReIDEmbedder: CLIP-based Re-ID (ViT backbone, 1280-D) — highest accuracy
  - OSNetEmbedder:    OSNet-based Re-ID (CNN backbone, 512-D)  — faster, lighter
  - OSNetONNXEmbedder: ONNX Runtime OSNet backend (512-D)      — no PyTorch inference

All implement the ReIDEmbedder protocol:
    embedder(crops: list[np.ndarray]) → list[list[float]]
    embedder.feat_dim → int
"""

import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.onnx_runtime import build_session


@runtime_checkable
class ReIDEmbedder(Protocol):
    """Protocol for Re-ID feature extractors."""

    @property
    def feat_dim(self) -> int: ...

    def __call__(self, crops: list[np.ndarray]) -> list[list[float]]: ...


_REID_H, _REID_W = 256, 128   # Standard person Re-ID input resolution
_MIN_H,  _MIN_W  = 32,  16    # Crops smaller than this produce garbage features
_PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_PIXEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _prepare_crop(bgr: np.ndarray) -> np.ndarray:
    """
    BGR → RGB + edge-pad if crop is smaller than _MIN_H × _MIN_W.
    Padding replicates border pixels so the model sees a plausible appearance.
    """
    rgb = bgr[..., ::-1].copy()
    h, w = rgb.shape[:2]
    if h < _MIN_H or w < _MIN_W:
        pad_h = max(0, _MIN_H - h)
        pad_w = max(0, _MIN_W - w)
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return rgb


def _is_onnx_path(path: str | None) -> bool:
    return bool(path) and Path(path).suffix.lower() == ".onnx"


def _make_batch_dynamic(weights_path: str) -> str:
    """
    Attempt to patch a fixed-batch ONNX model to support dynamic batching.

    Many OSNet ONNX exports use Reshape([1, -1]) before the FC layer, which
    forces batch_size=1.  Changing this to Reshape([-1, C]) makes the model
    accept any batch size with identical numerical output.

    Returns the path to the patched model (cached alongside the original),
    or the original path if patching is not needed or fails.
    """
    try:
        import onnx
    except ImportError:
        return weights_path

    original = Path(weights_path)
    cached = original.with_stem(original.stem + "_dynbatch")
    if cached.exists():
        return str(cached)

    try:
        model = onnx.load(str(original))
    except Exception:
        return weights_path

    # Check if input batch dim is already dynamic
    inp = model.graph.input[0]
    in_dims = inp.type.tensor_type.shape.dim
    if not in_dims or not isinstance(in_dims[0].dim_value, int) or in_dims[0].dim_value != 1:
        return weights_path  # already dynamic or unexpected shape

    # Find Reshape → Gemm pattern at the end of the network
    # Look for a Constant node that feeds a Reshape with shape [1, -1]
    node_by_output: dict[str, object] = {}
    for node in model.graph.node:
        for out_name in node.output:
            node_by_output[out_name] = node

    patched = False
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape_node = node_by_output.get(node.input[1])
        if shape_node is None or shape_node.op_type != "Constant":
            continue

        for attr in shape_node.attribute:
            if attr.name != "value":
                continue
            shape_data = np.frombuffer(attr.t.raw_data, dtype=np.int64)
            if len(shape_data) == 2 and shape_data[0] == 1 and shape_data[1] == -1:
                # Find the feature dim from the downstream Gemm weight
                consumers = [
                    n for n in model.graph.node
                    if any(node.output[0] == i for i in n.input)
                ]
                feat_dim = None
                for consumer in consumers:
                    if consumer.op_type == "Gemm":
                        for init in model.graph.initializer:
                            if init.name == consumer.input[1]:
                                feat_dim = int(init.dims[1])
                                break
                        break

                if feat_dim is not None:
                    new_shape = np.array([-1, feat_dim], dtype=np.int64)
                    attr.t.raw_data = new_shape.tobytes()
                    patched = True

    if not patched:
        return weights_path

    # Make input/output batch dimensions dynamic
    in_dims[0].dim_param = "batch"
    in_dims[0].ClearField("dim_value")

    out = model.graph.output[0]
    out_dims = out.type.tensor_type.shape.dim
    if out_dims and isinstance(out_dims[0].dim_value, int):
        out_dims[0].dim_param = "batch"
        out_dims[0].ClearField("dim_value")

    try:
        onnx.save(model, str(cached))
    except Exception:
        return weights_path

    return str(cached)


class CLIPReIDEmbedder:
    """
    CLIP-based Re-ID embedder using boxmot's ReID backend.

    Uses a Vision Transformer (ViT) pre-trained with CLIP then fine-tuned on
    person Re-ID datasets. Produces 1280-D L2-normalised embeddings.

    Args:
        weights: boxmot CLIP-ReID weight name (auto-downloaded).
        device: "auto" | "cuda" | "cpu" | "cuda:0" etc.
        half: FP16 inference (GPU only).
    """

    def __init__(
        self,
        weights: str = "clip_market1501.pt",
        device: str = "auto",
        half: bool = False,
    ):
        try:
            from boxmot import ReID
        except ImportError:
            raise ImportError(
                "boxmot is required for CLIP-ReID.\n"
                "Install: pip install boxmot"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not device.startswith("cuda"):
            half = False

        self.device = device
        self.half = half
        self._reid = ReID(weights=weights, device=device, half=half)
        self._feat_dim: int | None = None

    @property
    def feat_dim(self) -> int:
        return self._feat_dim or 1280

    @torch.no_grad()
    def __call__(self, crops: list) -> list:
        if not crops:
            return []

        max_w = max(c.shape[1] for c in crops)
        padded = []
        xyxy_list = []
        y_offset = 0
        for crop in crops:
            h, w = crop.shape[:2]
            if h < _MIN_H or w < _MIN_W:
                pad_h = max(0, _MIN_H - h)
                pad_w = max(0, _MIN_W - w)
                crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                h, w = crop.shape[:2]
            if w < max_w:
                crop = np.pad(crop, ((0, 0), (0, max_w - w), (0, 0)), mode="edge")
            padded.append(crop)
            xyxy_list.append([0, y_offset, w, y_offset + h])
            y_offset += h

        canvas = np.concatenate(padded, axis=0)
        xyxy = np.array(xyxy_list, dtype=np.float32)

        embs = self._reid.model.get_features(xyxy, canvas)
        if isinstance(embs, torch.Tensor):
            embs = F.normalize(embs.float(), dim=1)
            result = embs.cpu().numpy().tolist()
        else:
            embs = embs.astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            result = (embs / norms).tolist()

        if self._feat_dim is None and result:
            self._feat_dim = len(result[0])
        return result


class OSNetEmbedder:
    """
    OSNet-based Re-ID embedder compatible with deep-sort-realtime.

    Args:
        model_name: osnet_x0_25 / x0_5 / x0_75 / x1_0 / osnet_ain_x1_0
        weights_path: path to .pth weights; None = auto-download Market-1501
        device: "auto" | "cuda" | "cpu"
        tta: Test-Time Augmentation (h-flip averaging)
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        weights_path: str | None = None,
        device: str = "auto",
        half: bool = False,
        tta: bool = True,
    ):
        try:
            from torchreid.reid.utils import FeatureExtractor
        except ImportError:
            raise ImportError(
                "torchreid is required for OSNet Re-ID.\n"
                "Install: pip install torchreid gdown tensorboard"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not device.startswith("cuda"):
            half = False

        self.device = device
        self.half = half
        self.tta = tta

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._extractor = FeatureExtractor(
                model_name=model_name,
                model_path=weights_path or "",
                image_size=(_REID_H, _REID_W),
                device=device,
                verbose=False,
            )

        if half:
            self._extractor.model.half()

    @property
    def feat_dim(self) -> int:
        return 512

    def _extract_features(self, rgb_imgs: list[np.ndarray]) -> torch.Tensor:
        images = [
            self._extractor.preprocess(self._extractor.to_pil(img))
            for img in rgb_imgs
        ]
        images = torch.stack(images, dim=0).to(self._extractor.device)
        if self.half:
            images = images.half()
        return self._extractor.model(images)

    @torch.no_grad()
    def __call__(self, crops: list) -> list:
        if not crops:
            return []

        rgb_imgs = [_prepare_crop(c) for c in crops]
        feats = self._extract_features(rgb_imgs)

        if self.tta:
            rgb_flip = [img[:, ::-1, :].copy() for img in rgb_imgs]
            feats_flip = self._extract_features(rgb_flip)
            feats = (feats + feats_flip) * 0.5

        feats = F.normalize(feats.float(), dim=1)
        return feats.cpu().numpy().tolist()


class OSNetONNXEmbedder:
    """
    OSNet Re-ID embedder backed by ONNX Runtime.

    This is intended for exported OSNet `.onnx` models so the runtime path no
    longer depends on PyTorch inference.
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "auto",
        half: bool = False,
        tta: bool = True,
        compute_cfg: dict | None = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for OSNet ONNX Re-ID.\n"
                "Install: pip install onnxruntime-gpu   # or onnxruntime for CPU"
            ) from exc

        if not _is_onnx_path(weights_path):
            raise ValueError(f"Expected an .onnx Re-ID model, got: {weights_path!r}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.half = half and device.startswith("cuda")
        self.tta = tta

        # Attempt to enable dynamic batching via model surgery
        model_path = _make_batch_dynamic(weights_path)

        self._session, active_providers, self.provider_mode = build_session(
            ort,
            model_path,
            device=device,
            half=self.half,
            compute_cfg=compute_cfg,
            model_label="OSNet ONNX Re-ID",
        )
        self.providers = tuple(active_providers)
        if device.startswith("cuda") and not any(
            provider in self.providers for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
        ):
            warnings.warn(
                "OSNet ONNX Re-ID requested CUDA/TensorRT, but ONNX Runtime did not enable a GPU execution provider. "
                "Falling back to CPUExecutionProvider.",
                RuntimeWarning,
            )

        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        self._input_type = input_meta.type
        # Check if batch dimension is fixed (e.g., 1) or dynamic (str/None)
        in_shape = list(input_meta.shape)
        self._dynamic_batch = (
            not in_shape or not isinstance(in_shape[0], int) or in_shape[0] < 1
        )
        output_meta = self._session.get_outputs()[0]
        self._output_name = output_meta.name
        out_shape = list(output_meta.shape)
        self._feat_dim = int(out_shape[-1]) if out_shape and isinstance(out_shape[-1], int) else 512

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def _input_dtype(self) -> np.dtype:
        return np.float16 if self._input_type == "tensor(float16)" else np.float32

    def _preprocess_batch(self, rgb_imgs: list[np.ndarray]) -> np.ndarray:
        batch = []
        for img in rgb_imgs:
            resized = cv2.resize(img, (_REID_W, _REID_H), interpolation=cv2.INTER_LINEAR)
            arr = resized.astype(np.float32) / 255.0
            arr = (arr - _PIXEL_MEAN) / _PIXEL_STD
            arr = np.transpose(arr, (2, 0, 1))
            batch.append(arr)
        return np.ascontiguousarray(np.stack(batch, axis=0).astype(self._input_dtype(), copy=False))

    def _infer(self, rgb_imgs: list[np.ndarray]) -> torch.Tensor:
        if self._dynamic_batch:
            batch = self._preprocess_batch(rgb_imgs)
            outputs = self._session.run([self._output_name], {self._input_name: batch})[0]
            feats = np.asarray(outputs).reshape(len(rgb_imgs), -1)
        else:
            # Model has fixed batch=1 — run each image individually
            results = []
            for img in rgb_imgs:
                single = self._preprocess_batch([img])
                out = self._session.run([self._output_name], {self._input_name: single})[0]
                results.append(np.asarray(out).reshape(-1))
            feats = np.stack(results, axis=0) if results else np.empty((0, self._feat_dim), dtype=np.float32)
        return torch.from_numpy(feats.astype(np.float32, copy=False))

    @torch.no_grad()
    def __call__(self, crops: list) -> list:
        if not crops:
            return []

        rgb_imgs = [_prepare_crop(c) for c in crops]
        feats = self._infer(rgb_imgs)

        if self.tta:
            rgb_flip = [img[:, ::-1, :].copy() for img in rgb_imgs]
            feats_flip = self._infer(rgb_flip)
            feats = (feats + feats_flip) * 0.5

        feats = F.normalize(feats.float(), dim=1)
        return feats.cpu().numpy().tolist()
