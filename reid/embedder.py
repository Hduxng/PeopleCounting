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
        self._session, active_providers, self.provider_mode = build_session(
            ort,
            weights_path,
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
        batch = self._preprocess_batch(rgb_imgs)
        outputs = self._session.run([self._output_name], {self._input_name: batch})[0]
        feats = np.asarray(outputs).reshape(len(rgb_imgs), -1)
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
