#!/usr/bin/env python3
"""Export detection or Re-ID checkpoints to ONNX."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def _load_checkpoint(source: Path):
    try:
        import torch
    except ImportError:
        return None

    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except Exception:
        return None


def _inspect_pytorch_checkpoint_kind(source: Path) -> str | None:
    ckpt = _load_checkpoint(source)
    if not isinstance(ckpt, dict):
        return None

    args = ckpt.get("args")
    pretrain_weights = str(getattr(args, "pretrain_weights", "")).lower() if args is not None else ""
    state = ckpt.get("model")
    if hasattr(state, "keys"):
        keys = list(state.keys())
    else:
        keys = list(ckpt.keys())

    if "rf-detr" in pretrain_weights:
        return "rf-detr"

    detector_markers = (
        "transformer.decoder.layers.0",
        "transformer.enc_out_class_embed",
        "transformer.enc_out_bbox_embed",
        "backbone.0.encoder.encoder",
    )
    if any(any(marker in key for marker in detector_markers) for key in keys):
        return "rf-detr"

    osnet_markers = (
        "conv1.weight",
        "classifier.weight",
        "fc.0.weight",
    )
    if any(any(marker in key for marker in osnet_markers) for key in keys):
        return "osnet"

    return None


def _infer_rfdetr_variant(source: Path, override: str) -> str:
    if override != "auto":
        return override

    ckpt = _load_checkpoint(source)
    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    token = str(getattr(args, "pretrain_weights", "")).lower() if args is not None else ""

    for variant in ("base", "nano", "small", "medium", "large"):
        if f"rf-detr-{variant}" in token:
            return variant

    resolution = getattr(args, "resolution", None) if args is not None else None
    if resolution is not None:
        if int(resolution) >= 704:
            return "large"
        if int(resolution) >= 576:
            return "medium"
        if int(resolution) >= 512:
            return "small"
        if int(resolution) >= 384:
            return "nano"

    raise SystemExit(
        f"Cannot infer RF-DETR variant from {source.name!r}. "
        "Pass --rfdetr-variant base|nano|small|medium|large."
    )


def _infer_kind(source: Path, kind: str) -> str:
    if kind != "auto":
        return kind

    suffix = source.suffix.lower()
    if suffix == ".pt":
        return "yolo"
    if suffix == ".pth":
        detected = _inspect_pytorch_checkpoint_kind(source)
        if detected == "osnet":
            return "osnet"
        if detected == "rf-detr":
            return "rfdetr"
        raise SystemExit(
            f"Cannot safely infer exporter kind from {source.name!r}. "
            "Use --kind yolo, --kind osnet, or --kind rfdetr."
        )
    raise SystemExit(
        f"Cannot infer exporter kind from {source.name!r}. "
        "Use --kind yolo, --kind osnet, or --kind rfdetr."
    )


def _resolve_output_path(source: Path, output: str | None) -> Path:
    target = Path(output) if output else source.with_suffix(".onnx")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _resolve_imgsz(values: list[int]) -> int | list[int]:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values
    raise SystemExit("--imgsz expects 1 or 2 integers")


def _resolve_device(raw: str) -> str:
    if raw != "auto":
        return raw
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_rfdetr_num_classes(source: Path, override: int | None) -> int:
    if override is not None:
        return override

    ckpt = _load_checkpoint(source)
    if not isinstance(ckpt, dict):
        return 1

    args = ckpt.get("args")
    class_names = list(getattr(args, "class_names", []) or []) if args is not None else []
    if class_names:
        return len(class_names)

    model_state = ckpt.get("model", {})
    bias = model_state.get("class_embed.bias") if hasattr(model_state, "get") else None
    if bias is not None and hasattr(bias, "shape") and len(bias.shape) == 1:
        return max(int(bias.shape[0]) - 1, 1)

    return 1


def _resolve_rfdetr_resolution(source: Path, override: int | None) -> int | None:
    if override is not None:
        return override

    ckpt = _load_checkpoint(source)
    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    resolution = getattr(args, "resolution", None) if args is not None else None
    return int(resolution) if resolution is not None else None


def _resolve_rfdetr_shape(source: Path, override: list[int] | None, resolution: int | None) -> tuple[int, int] | None:
    if override:
        if len(override) != 2:
            raise SystemExit("--rfdetr-shape expects exactly 2 integers: HEIGHT WIDTH")
        return int(override[0]), int(override[1])
    if resolution is not None:
        return int(resolution), int(resolution)
    return None


def _export_yolo(args, source: Path, output: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required for YOLO ONNX export") from exc

    device = _resolve_device(args.device)
    model = YOLO(str(source))
    export_path = Path(model.export(
        format="onnx",
        imgsz=_resolve_imgsz(args.imgsz),
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        device=device,
        batch=args.batch_size,
    ))

    if export_path.resolve() != output.resolve():
        if output.exists():
            output.unlink()
        shutil.move(str(export_path), str(output))
    return output


def _export_osnet(args, source: Path, output: Path) -> Path:
    try:
        import torch
        from torchreid.reid.models import build_model
        from torchreid.reid.utils import load_pretrained_weights
    except ImportError as exc:
        raise SystemExit("torchreid and torch are required for OSNet ONNX export") from exc

    device_str = _resolve_device(args.device)
    device = torch.device(device_str)
    if args.half and device.type != "cuda":
        raise SystemExit("--half export for OSNet requires a CUDA device")

    model = build_model(
        name=args.osnet_model,
        num_classes=1,
        loss="softmax",
        pretrained=not source.exists(),
        use_gpu=device.type == "cuda",
    )
    if source.exists():
        load_pretrained_weights(model, str(source))

    model.eval().to(device)
    dummy = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    if args.half:
        model.half()
        dummy = dummy.half()

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "images": {0: "batch"},
            "embeddings": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["embeddings"],
            dynamic_axes=dynamic_axes,
        )
    return output


def _export_rfdetr(args, source: Path, output: Path) -> Path:
    try:
        import rfdetr
    except ImportError as exc:
        raise SystemExit(
            "RF-DETR export requires the official rfdetr package. Install: pip install rfdetr"
        ) from exc

    variant = _infer_rfdetr_variant(source, args.rfdetr_variant)
    variant_map = {
        "base": rfdetr.RFDETRBase,
        "nano": rfdetr.RFDETRNano,
        "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium,
        "large": rfdetr.RFDETRLarge,
    }
    cls = variant_map[variant]

    resolution = _resolve_rfdetr_resolution(source, args.rfdetr_resolution)
    shape = _resolve_rfdetr_shape(source, args.rfdetr_shape, resolution)
    num_classes = _resolve_rfdetr_num_classes(source, args.rfdetr_num_classes)
    device = _resolve_device(args.device)

    init_kwargs = {
        "pretrain_weights": str(source),
        "num_classes": num_classes,
        "device": device,
    }
    if resolution is not None:
        init_kwargs["resolution"] = resolution

    model = cls(**init_kwargs)

    with tempfile.TemporaryDirectory(prefix="rfdetr_export_") as tmp_dir:
        model.export(
            output_dir=tmp_dir,
            shape=shape,
            batch_size=args.batch_size,
            dynamic_batch=args.dynamic,
            opset_version=args.opset,
            verbose=not args.quiet,
        )
        exported = Path(tmp_dir) / "inference_model.onnx"
        if not exported.exists():
            raise SystemExit(f"RF-DETR export did not produce expected file: {exported}")
        if output.exists():
            output.unlink()
        shutil.move(str(exported), str(output))

    return output


def _validate_onnx(path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("[warn] onnx package not installed; skipping ONNX validation")
        return

    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Input .pt or .pth file")
    parser.add_argument("-o", "--output", help="Output ONNX path")
    parser.add_argument("--kind", choices=["auto", "yolo", "osnet", "rfdetr"], default="auto")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    parser.add_argument("--half", action="store_true", help="Export FP16 model when supported")
    parser.add_argument("--dynamic", action="store_true", help="Export dynamic batch axis")
    parser.add_argument("--simplify", action="store_true", help="Run YOLO graph simplification when available")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy/export batch size")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="YOLO export image size: 640 or 640 640")
    parser.add_argument("--osnet-model", default="osnet_x1_0", help="OSNet architecture name for .pth export")
    parser.add_argument("--height", type=int, default=256, help="OSNet input height")
    parser.add_argument("--width", type=int, default=128, help="OSNet input width")
    parser.add_argument("--rfdetr-variant", choices=["auto", "base", "nano", "small", "medium", "large"], default="auto", help="RF-DETR variant for .pth detector checkpoints")
    parser.add_argument("--rfdetr-num-classes", type=int, default=None, help="Override RF-DETR num_classes during export")
    parser.add_argument("--rfdetr-resolution", type=int, default=None, help="Override RF-DETR square resolution during export")
    parser.add_argument("--rfdetr-shape", nargs=2, type=int, default=None, help="Override RF-DETR export shape: HEIGHT WIDTH")
    parser.add_argument("--quiet", action="store_true", help="Reduce exporter verbosity when supported")
    parser.add_argument("--skip-check", action="store_true", help="Skip ONNX checker validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Input file does not exist: {source}")

    kind = _infer_kind(source, args.kind)
    output = _resolve_output_path(source, args.output)

    if kind == "yolo":
        exported = _export_yolo(args, source, output)
    elif kind == "osnet":
        exported = _export_osnet(args, source, output)
    elif kind == "rfdetr":
        exported = _export_rfdetr(args, source, output)
    else:
        raise SystemExit(f"Unsupported exporter kind: {kind}")

    if not args.skip_check:
        _validate_onnx(exported)

    print(f"[ok] exported {kind} model -> {exported}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
