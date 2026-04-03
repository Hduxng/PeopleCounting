# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run on video file
python main.py --source video.mp4 --display

# Run on webcam
python main.py --source 0 --display

# Save output (default: output/<name>_<timestamp>.mp4)
python main.py --source video.mp4
python main.py --source video.mp4 --save output.mp4
python main.py --no-save --display

# Calibrate crossline direction interactively
python tools/calibrate_line.py --source video.mp4

# Tests
pytest tests/ -v                       # All tests
pytest tests/test_counters.py -v       # Counter logic (no GPU needed)
pytest tests/test_reid.py -v           # Re-ID quality (requires CUDA)
pytest tests/test_tracking.py -v       # Tracker integration (requires CUDA)
pytest tests/test_anchor.py -v         # Anchor injection (no GPU needed)
pytest tests/test_validation.py -v     # Bbox validation (no GPU needed)

# Install dependencies
pip install -r requirements.txt
```

## Architecture

**Pipeline (per frame, orchestrated in `main.py` `run()`):**

```
YOLO Detection → Anchor Injection → Re-ID Embedding (batch GPU) →
Tracker (BoT-SORT/StrongSORT/DeepSORT/ByteTrack) → Gallery Recovery →
Bbox Smoothing → Validation → Counting (Crossline & Zone) → Visualization
```

### Key modules

- **`main.py`** — Entry point. Wires together all components and runs the frame loop. Handles CLI args, config loading, tracker initialization, and the per-frame pipeline.
- **`counter/base.py`** — Abstract `BaseCounter` interface.
- **`counter/crossline.py`** — Line-crossing counter with spatial hysteresis (buffer zone) and temporal hysteresis (N consecutive confirm frames). Uses cross-product for side detection.
- **`counter/zone.py`** — Polygon zone counter with dwell-time threshold and refractory period to prevent boundary jitter double-counts.
- **`reid/embedder.py`** — `CLIPReIDEmbedder` (ViT, 1280-D) and `OSNetEmbedder` (512-D). All crops embedded in a single batch GPU forward pass per frame.
- **`reid/gallery.py`** — `TrackGallery`: per-track EMA feature storage, recovers lost track IDs by cosine distance matching. Includes drift detection for merge-split scenarios.
- **`reid/histogram_gallery.py`** — Lightweight alternative gallery for ByteTrack using color histograms.
- **`utils/anchor.py`** — `TrackAnchor`: injects velocity-extrapolated synthetic detections when YOLO misses confirmed tracks. Confidence set below `confidence_new_track` to avoid spurious counts.
- **`utils/geometry.py`** — Point-in-polygon (OpenCV), cross-product line-side detection.
- **`utils/validation.py`** — `TrackValidator`: soft-gates tracks by penalizing bbox aspect-ratio/area inconsistencies.
- **`utils/drawing.py`** — OpenCV visualization (bboxes, IDs, lines, zones, counts).
- **`utils/cmc.py`** — Camera motion compensation (disabled by default).

### Dual-confidence strategy

`detector.confidence` (0.25) is the YOLO threshold — kept low to catch occluded people. `detector.confidence_new_track` (0.45) is the minimum to spawn a new track. Counters only fire on detections at or above the new-track threshold.

### Counter state & Re-ID remaps

When the gallery recovers a lost track ID, counter state is transferred to the recovered ID via `transfer_id()`. Lost tracks retain counter state for `retention_frames` (derived from tracker buffer + gallery lifetime).

## Configuration

All tunable parameters live in `config.yaml`. Key sections: `video`, `compute`, `detector`, `tracker` (type, reid, gallery, anchor, histogram_gallery), `counting` (mode, crosslines, zones). Tracker type is one of: `botsort`, `strongsort`, `deepsort`, `bytetrack`.

## Testing

Tests use a `SceneSimulator` fixture (in `tests/conftest.py`) that generates synthetic frames with colored moving rectangles, bypassing YOLO for isolated unit testing. GPU-dependent tests are skipped when CUDA is unavailable.
