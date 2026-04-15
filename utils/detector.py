"""Detector backends and ONNX helpers."""

from __future__ import annotations

from pathlib import Path
import warnings

import cv2
import numpy as np

from utils.onnx_runtime import build_session


_RFDETR_PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RFDETR_PIXEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


_DEFAULT_DEDUP_CFG = {
    "enabled": True,
    "iou": 0.68,
    "containment": 0.82,
    "area_ratio": 0.55,
    "center_ratio": 0.24,
}


def _resolve_dedup_cfg(dedup_cfg: dict | None) -> dict:
    cfg = dict(_DEFAULT_DEDUP_CFG)
    if dedup_cfg:
        cfg.update(dedup_cfg)
    return cfg


def deduplicate_detections(dets: np.ndarray, dedup_cfg: dict | None = None) -> np.ndarray:
    dets = np.asarray(dets, dtype=np.float32)
    if dets.size == 0 or len(dets) < 2:
        return dets.astype(np.float32, copy=False)

    cfg = _resolve_dedup_cfg(dedup_cfg)
    if not cfg.get("enabled", True):
        return dets.astype(np.float32, copy=False)

    order = np.argsort(dets[:, 4], kind="stable")[::-1]
    keep_indices: list[int] = []
    iou_thresh = float(cfg.get("iou", _DEFAULT_DEDUP_CFG["iou"]))
    containment_thresh = float(cfg.get("containment", _DEFAULT_DEDUP_CFG["containment"]))
    area_ratio_thresh = float(cfg.get("area_ratio", _DEFAULT_DEDUP_CFG["area_ratio"]))
    center_ratio_thresh = float(cfg.get("center_ratio", _DEFAULT_DEDUP_CFG["center_ratio"]))

    for idx in order:
        cand = dets[idx]
        cx1, cy1, cx2, cy2 = cand[:4]
        cand_area = max(float((cx2 - cx1) * (cy2 - cy1)), 1.0)
        cand_cls = cand[5] if dets.shape[1] > 5 else None
        suppress = False

        for kept_idx in keep_indices:
            kept = dets[kept_idx]
            if cand_cls is not None and dets.shape[1] > 5 and kept[5] != cand_cls:
                continue

            kx1, ky1, kx2, ky2 = kept[:4]
            ix1 = max(float(cx1), float(kx1))
            iy1 = max(float(cy1), float(ky1))
            ix2 = min(float(cx2), float(kx2))
            iy2 = min(float(cy2), float(ky2))
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            inter = float((ix2 - ix1) * (iy2 - iy1))
            kept_area = max(float((kx2 - kx1) * (ky2 - ky1)), 1.0)
            union = cand_area + kept_area - inter
            iou = inter / union if union > 0 else 0.0
            smaller_area = min(cand_area, kept_area)
            area_ratio = smaller_area / max(cand_area, kept_area)
            inter_over_smaller = inter / max(smaller_area, 1.0)

            cand_center = np.array([(cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5], dtype=np.float32)
            kept_center = np.array([(kx1 + kx2) * 0.5, (ky1 + ky2) * 0.5], dtype=np.float32)
            center_dist = float(np.linalg.norm(cand_center - kept_center))
            if cand_area <= kept_area:
                diag_ref = max(float(np.hypot(cx2 - cx1, cy2 - cy1)), 1.0)
            else:
                diag_ref = max(float(np.hypot(kx2 - kx1, ky2 - ky1)), 1.0)
            aligned = center_dist <= center_ratio_thresh * diag_ref

            if aligned and (
                (iou >= iou_thresh and area_ratio >= area_ratio_thresh)
                or inter_over_smaller >= containment_thresh
            ):
                suppress = True
                break

        if not suppress:
            keep_indices.append(int(idx))

    return dets[np.asarray(keep_indices, dtype=np.int64)].astype(np.float32, copy=False)


def is_rfdetr_onnx_model(path: str | Path) -> bool:
    model_path = Path(path)
    try:
        import onnx
    except ImportError:
        return False

    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception:
        return False

    output_names = {output.name for output in model.graph.output}
    return {"dets", "labels"}.issubset(output_names)


class RFDETRONNXDetector:
    """RF-DETR detector backed by ONNX Runtime."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        half: bool = False,
        compute_cfg: dict | None = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for RF-DETR ONNX inference. "
                "Install onnxruntime-gpu or onnxruntime."
            ) from exc

        self.weights_path = str(weights_path)
        self.device = device
        self._ort = ort
        self._session, active_providers, self.provider_mode = build_session(
            ort,
            self.weights_path,
            device=device,
            half=half,
            compute_cfg=compute_cfg,
            model_label="RF-DETR ONNX",
        )
        self.providers = tuple(active_providers)

        if device.startswith("cuda") and not any(
            provider in self.providers for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
        ):
            warnings.warn(
                "RF-DETR ONNX requested CUDA/TensorRT, but ONNX Runtime did not enable a GPU execution provider. "
                "Falling back to CPUExecutionProvider.",
                RuntimeWarning,
            )

        input_meta = self._session.get_inputs()[0]
        input_shape = list(input_meta.shape)
        if len(input_shape) != 4:
            raise ValueError(f"Unsupported RF-DETR ONNX input shape: {input_shape}")
        _, channels, height, width = input_shape
        if channels != 3:
            raise ValueError(f"Unsupported RF-DETR ONNX channel count: {channels}")
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(
                "RF-DETR ONNX export must have static input height/width for this runtime path."
            )

        outputs = {meta.name: meta for meta in self._session.get_outputs()}
        if not {"dets", "labels"}.issubset(outputs):
            raise ValueError(
                f"ONNX graph {weights_path!r} is not an RF-DETR detector. "
                f"Expected outputs dets/labels, got {sorted(outputs)}"
            )

        self._input_name = input_meta.name
        self._input_type = input_meta.type
        self._input_h = int(height)
        self._input_w = int(width)
        self._boxes_name = "dets"
        self._logits_name = "labels"
        label_shape = list(outputs[self._logits_name].shape)
        self._label_dim = int(label_shape[-1]) if label_shape and isinstance(label_shape[-1], int) else None
        self._num_select = 300

    def _input_dtype(self) -> np.dtype:
        return np.float16 if self._input_type == "tensor(float16)" else np.float32

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - _RFDETR_PIXEL_MEAN) / _RFDETR_PIXEL_STD
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor.astype(self._input_dtype(), copy=False))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float32)
        cx = boxes[..., 0]
        cy = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]
        return np.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), axis=-1)

    def _postprocess(
        self,
        dets: np.ndarray,
        logits: np.ndarray,
        frame_shape: tuple[int, int],
        conf: float,
        classes: list[int] | tuple[int, ...] | None,
        dedup_cfg: dict | None = None,
    ) -> np.ndarray:
        boxes = self._box_cxcywh_to_xyxy(np.asarray(dets, dtype=np.float32)[0])
        out_logits = np.asarray(logits, dtype=np.float32)[0]
        scores = self._sigmoid(out_logits).reshape(-1)
        if scores.size == 0 or boxes.size == 0:
            return np.empty((0, 6), dtype=np.float32)

        topk = min(self._num_select, scores.size)
        order = np.argsort(scores)[::-1][:topk]
        topk_scores = scores[order]
        num_labels = out_logits.shape[1]
        box_indices = order // num_labels
        label_indices = order % num_labels
        boxes = boxes[box_indices]

        frame_h, frame_w = frame_shape
        scale = np.array([frame_w, frame_h, frame_w, frame_h], dtype=np.float32)
        boxes = boxes * scale
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, float(frame_w))
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, float(frame_h))

        keep = topk_scores >= float(conf)
        if self._label_dim is not None and self._label_dim > 1:
            keep &= label_indices != (self._label_dim - 1)
        if classes is not None:
            class_ids = np.array(list(classes), dtype=np.int64)
            keep &= np.isin(label_indices, class_ids)
        keep &= boxes[:, 2] > boxes[:, 0]
        keep &= boxes[:, 3] > boxes[:, 1]

        if not np.any(keep):
            return np.empty((0, 6), dtype=np.float32)

        boxes = boxes[keep]
        topk_scores = topk_scores[keep]
        label_indices = label_indices[keep].astype(np.float32, copy=False)
        dets_np = np.concatenate(
            [boxes.astype(np.float32, copy=False), topk_scores[:, None], label_indices[:, None]],
            axis=1,
        )
        return deduplicate_detections(dets_np, dedup_cfg=dedup_cfg)

    def predict(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        classes: list[int] | tuple[int, ...] | None = None,
        dedup_cfg: dict | None = None,
    ) -> np.ndarray:
        batch = self._preprocess(frame)
        dets, logits = self._session.run(
            [self._boxes_name, self._logits_name],
            {self._input_name: batch},
        )
        return self._postprocess(
            dets,
            logits,
            frame.shape[:2],
            conf=conf,
            classes=classes,
            dedup_cfg=dedup_cfg,
        )
