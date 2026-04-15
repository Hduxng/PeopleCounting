# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run on video file (auto-saves to output/)
python main.py --source video.mp4

# With live preview
python main.py --source video.mp4 --display

# Webcam / RTSP
python main.py --source 0 --display
python main.py --source "rtsp://user:pass@host:554/stream"

# Custom output / no save
python main.py --source video.mp4 --save output/custom.mp4
python main.py --source video.mp4 --no-save --display

# Calibrate crossline direction interactively
python tools/calibrate_line.py --source video.mp4

# Tests
pytest tests/ -v                         # all tests
pytest tests/test_counters.py -v         # counter logic (no GPU)
pytest tests/test_anchor.py -v           # anchor injection (no GPU)
pytest tests/test_validation.py -v       # bbox validation (no GPU)
pytest tests/test_reid.py -v             # Re-ID quality (CUDA required)
pytest tests/test_tracking.py -v         # tracker integration (CUDA required)
pytest tests/test_counters.py::test_name -v  # single test

# Install
pip install -r requirements.txt
```

## Architecture

**Per-frame pipeline (orchestrated in `main.py` `run()`):**

```
Video → Detection (YOLO or RF-DETR ONNX) → Deduplication (RF-DETR only)
      → Anchor Injection → Re-ID Embedding (batch GPU)
      → Tracker (nwojke/BoT-SORT/StrongSORT/DeepSORT/ByteTrack)
      → Gallery Recovery → Bbox Smoothing → Validation
      → Counting (Crossline / Zone) → Visualization → Output
```

### Key modules

- **`main.py`** — Entry point. Wires all components, runs frame loop. Handles CLI args, config loading, tracker init, and per-frame pipeline.
- **`counter/base.py`** — Abstract `BaseCounter` with `transfer_id()` for Re-ID remaps.
- **`counter/crossline.py`** — Line-crossing counter with spatial hysteresis (buffer zone) and temporal hysteresis (N confirm frames). Uses cross-product for side detection.
- **`counter/zone.py`** — Polygon zone counter with dwell-time threshold and refractory period to prevent boundary jitter double-counts.
- **`reid/embedder.py`** — `CLIPReIDEmbedder` (ViT, 1280-D) and `OSNetEmbedder` (512-D). Single batch GPU forward pass per frame.
- **`reid/gallery.py`** — `TrackGallery`: per-track EMA feature storage with prototype bank, recovers lost track IDs by cosine distance + Hungarian matching. Includes merge-split drift detection (freezes features when bbox area spikes).
- **`reid/histogram_gallery.py`** — Lightweight alternative for ByteTrack using color histograms (no model needed).
- **`utils/detector.py`** — `RFDETRONNXDetector` for RF-DETR ONNX inference with built-in deduplication. `is_rfdetr_onnx_model()` to auto-detect model type.
- **`utils/onnx_runtime.py`** — Shared ONNX Runtime session builder with TensorRT → CUDA → CPU fallback chain.
- **`utils/anchor.py`** — `TrackAnchor`: injects velocity-extrapolated synthetic detections when detector misses confirmed tracks. Synthetic confidence set below `confidence_new_track` to avoid spurious new tracks.
- **`utils/geometry.py`** — Point-in-polygon (OpenCV), cross-product line-side detection.
- **`utils/validation.py`** — `TrackValidator`: soft-gates tracks by penalizing bbox aspect-ratio/area inconsistencies.
- **`utils/drawing.py`** — OpenCV visualization (bboxes, IDs, lines, zones, counts).
- **`deep_sort/`** — Vendored nwojke Deep SORT implementation (Kalman filter, Hungarian assignment, cosine metric).
- **`debug.py`** — `DebugLogger`: writes per-frame CSV (`tracks.csv`) and event log to `debug/<timestamp>/`. Enabled via `runtime.debug: true` in config.

### Key design decisions

**Dual-confidence strategy:** `detector.confidence` (0.25) is the YOLO/RF-DETR threshold — kept low to catch occluded people. `detector.confidence_new_track` (0.45) gates new track creation. Counters only fire on detections at or above the new-track threshold.

**Counter state & Re-ID remaps:** When the gallery recovers a lost track ID, counter state is transferred via `transfer_id()`. Lost tracks retain counter state for `retention_frames` (derived from tracker buffer + gallery lifetime).

**Detector backends:** YOLO models (`.pt`/`.onnx`) use ultralytics. RF-DETR ONNX models use custom `RFDETRONNXDetector` with `deduplicate_detections()` post-processing (IoU + containment + area ratio suppression).

**Tracker types:** `nwojke` is the vendored Deep SORT in `deep_sort/`. `botsort`/`strongsort`/`deepsort`/`bytetrack` use boxmot.

## Configuration

All tunable parameters live in `config.yaml`. Key sections:
- `video` — source, fps_override
- `compute` — device, half precision, ONNX Runtime providers (TensorRT/CUDA/CPU)
- `detector` — model path, confidence thresholds, NMS IoU, dedup settings (RF-DETR)
- `tracker` — type, reid config, gallery/histogram_gallery, anchor, post_validation, cmc
- `counting` — mode (line/zone/both), crosslines, zones
- `display` — visualization options, colors, smoothing
- `runtime` — display flag, debug logging
- `output` — save flag, path

## Testing

Tests use a `SceneSimulator` fixture (in `tests/conftest.py`) that generates synthetic frames with colored moving rectangles, bypassing real detection for isolated unit testing. GPU-dependent tests are skipped when CUDA is unavailable.
