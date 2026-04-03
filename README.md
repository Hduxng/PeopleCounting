# People Counting System

Video-based people counting using YOLO11x detection, BoT-SORT tracking, and CLIP Re-ID.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run on video file (output saved to output/ automatically)
python main.py --source video.mp4

# 3. View result
# Output: output/video_20260401_143022.mp4
```

## Commands

### Basic Usage

```bash
# Process video file (default: no preview, auto-save to output/)
python main.py --source video.mp4

# With live preview window
python main.py --source video.mp4 --display

# Webcam
python main.py --source 0 --display

# RTSP stream
python main.py --source "rtsp://user:pass@192.168.1.100:554/stream"

# Custom output path
python main.py --source video.mp4 --save output/custom_name.mp4

# No save (processing only)
python main.py --source video.mp4 --no-save --display

# Custom config
python main.py --config my_config.yaml --source video.mp4
```

### Tools

```bash
# Calibrate crossline direction (IN/OUT)
python tools/calibrate_line.py --source video.mp4

# Controls: Click=test side, F=flip IN/OUT, P=pick new line, S=save, Q=quit
```

### Tests

```bash
# Run all tests
pytest tests/ -v

# Counter logic only (fast, no GPU needed)
pytest tests/test_counters.py -v

# Re-ID embedding quality
pytest tests/test_reid.py -v

# Tracker integration (needs GPU, ~2 min)
pytest tests/test_tracking.py -v
```

## Architecture

```
Video → YOLO11x Detection → Anchor Injection → CLIP Re-ID → BoT-SORT Tracking
      → Gallery Recovery → Bbox Smoothing → Counting (Crossline / Zone) → Output
```

| Component | Model / Algorithm | Details |
|---|---|---|
| Detection | YOLO11x (ultralytics) | 110MB, 56.1 mAP COCO, FP16 |
| Tracking | BoT-SORT (boxmot) | Kalman + CMC + Re-ID, SOTA MOT17/MOT20 |
| Re-ID | CLIP-ReID ViT (boxmot) | 507MB, 1280-D, 93.8% mAP Market-1501 |
| Re-ID fallback | OSNet AIN x1.0 (torchreid) | 20MB, 512-D |
| Counting | Crossline + Zone | Cross-product + point-in-polygon |

## Configuration

All settings are in `config.yaml`.

### Detection

```yaml
detector:
  model: "yolo11x.pt"        # yolo11n/s/m/l/x.pt
  confidence: 0.25            # detection threshold
  confidence_new_track: 0.45  # min confidence to create new track
  iou: 0.65                   # NMS IoU (higher = keep more overlapping boxes)
  classes: [0]                # COCO class 0 = person
```

### Tracker

```yaml
tracker:
  type: "botsort"          # botsort | strongsort | deepsort | bytetrack

  # BoT-SORT params
  track_buffer: 120        # frames to keep lost track (~4s at 30fps)
  appearance_thresh: 0.25  # Re-ID matching threshold
  cmc_method: "ecc"        # Camera Motion Compensation: ecc | orb | sof | none
```

### Re-ID

```yaml
tracker:
  reid:
    type: "clip"                    # clip | osnet
    weights: "clip_market1501.pt"   # auto-downloaded on first run
```

### Counting Modes

```yaml
counting:
  mode: "line"    # line | zone | both
```

**Crossline** - count people crossing a virtual line:

```yaml
  crosslines:
    - id: "entrance"
      name: "Entrance"
      points: [[x1, y1], [x2, y2]]       # line endpoints
      enter_direction: "positive"          # positive | negative
      buffer_px: 20                        # hysteresis buffer
```

Use `python tools/calibrate_line.py` to visually determine `enter_direction`.

**Zone** - count people dwelling in a polygon:

```yaml
  zones:
    - id: "waiting"
      name: "Waiting Area"
      points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]  # polygon vertices
      min_dwell_seconds: 10.0                           # minimum stay time
```

### Display

```yaml
display:
  show_bbox: true
  show_ids: true
  show_centers: true
  bbox_ema_alpha: 0.85    # 1.0=no smoothing, lower=more smoothing
```

## Output

When running without `--display`, the console shows:

```
[progress] 500/1200 (42%)  28.3 fps  ETA 25s
[done] 1200 frames in 42.3s (28.4 fps)
  Entrance: IN=15  OUT=8
  Waiting Area: IN=6
[save] output saved: output/video_20260401_143022.mp4
```

## Project Structure

```
peopleCounting/
├── main.py              # Entry point
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
├── reid/
│   ├── embedder.py      # CLIPReIDEmbedder + OSNetEmbedder
│   └── gallery.py       # Re-ID recovery gallery
├── counter/
│   ├── base.py          # BaseCounter ABC
│   ├── crossline.py     # Line crossing counter
│   └── zone.py          # Zone dwell counter
├── utils/
│   ├── geometry.py      # Side-of-line, point-in-polygon
│   ├── drawing.py       # OpenCV visualization
│   └── anchor.py        # Synthetic detection injection
├── tools/
│   └── calibrate_line.py  # Crossline calibration tool
├── tests/
│   ├── test_counters.py   # 17 counter unit tests
│   ├── test_reid.py       # 22 Re-ID quality tests
│   └── test_tracking.py   # 18 tracker integration tests
└── output/              # Video results (auto-generated)
```

## Requirements

- Python >= 3.10
- CUDA GPU (for tracker + Re-ID inference)
- ~800MB disk for model weights (auto-downloaded on first run)

```bash
pip install -r requirements.txt
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Bbox drift when people stand close | Increase `detector.iou` (less aggressive NMS) |
| ID switches after occlusion | Increase `tracker.track_buffer` and `gallery.lifetime` |
| Missing small/far people | Lower `detector.confidence` |
| Slow processing | Use smaller model (`yolo11s.pt`), or set `reid.type: "osnet"` |
| No GPU available | Set `compute.device: "cpu"` and `compute.half: false` |
| Wrong IN/OUT direction | Run `python tools/calibrate_line.py` and press F to flip |
