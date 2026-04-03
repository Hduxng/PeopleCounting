"""
Crossline direction calibration tool.

Opens the first frame of a video, draws the crossline, and shows
which side is IN (enter) and which side is OUT (exit).

Click anywhere on the frame to see which side that point is on.

Usage:
    python tools/calibrate_line.py                          # uses config.yaml
    python tools/calibrate_line.py --source test.mp4        # override source
    python tools/calibrate_line.py --config my.yaml         # custom config

Controls:
    Click      — test which side a point is on
    F          — flip enter_direction (swap IN/OUT)
    S          — save current config to file
    P          — enter point-picking mode (click 2 points to redefine line)
    Q / ESC    — quit
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.geometry import point_side_of_line


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State:
    def __init__(self):
        self.frame = None
        self.pt1 = None
        self.pt2 = None
        self.enter_direction = "positive"
        self.buffer_px = 20
        self.last_click = None
        self.last_side = None
        self.picking_mode = False
        self.picked_points = []
        self.line_cfg_index = 0
        self.config_path = "config.yaml"


state = State()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_overlay(frame):
    vis = frame.copy()
    pt1, pt2 = state.pt1, state.pt2

    if pt1 is None or pt2 is None:
        cv2.putText(vis, "No crossline configured", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return vis

    pt1i = (int(pt1[0]), int(pt1[1]))
    pt2i = (int(pt2[0]), int(pt2[1]))

    # Draw line
    cv2.line(vis, pt1i, pt2i, (0, 255, 0), 3)
    cv2.circle(vis, pt1i, 8, (255, 255, 0), -1)
    cv2.circle(vis, pt2i, 8, (255, 255, 0), -1)
    cv2.putText(vis, "P1", (pt1i[0] + 10, pt1i[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(vis, "P2", (pt2i[0] + 10, pt2i[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Compute perpendicular direction (points to positive side)
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    line_len = math.sqrt(dx * dx + dy * dy)
    if line_len < 1:
        return vis

    # Normal vector pointing to positive side: (-dy, dx) normalized
    nx = -dy / line_len
    ny = dx / line_len

    # Midpoint
    mx = (pt1[0] + pt2[0]) / 2
    my = (pt1[1] + pt2[1]) / 2

    # Arrow offset
    offset = 60

    # Positive side arrow
    pos_x = int(mx + nx * offset)
    pos_y = int(my + ny * offset)
    # Negative side arrow
    neg_x = int(mx - nx * offset)
    neg_y = int(my - ny * offset)

    if state.enter_direction == "positive":
        in_pt, out_pt = (pos_x, pos_y), (neg_x, neg_y)
    else:
        in_pt, out_pt = (neg_x, neg_y), (pos_x, pos_y)

    # IN side (green)
    cv2.arrowedLine(vis, (int(mx), int(my)), in_pt, (0, 255, 0), 3, tipLength=0.3)
    cv2.putText(vis, "IN", (in_pt[0] - 15, in_pt[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # OUT side (red)
    cv2.arrowedLine(vis, (int(mx), int(my)), out_pt, (0, 0, 255), 3, tipLength=0.3)
    cv2.putText(vis, "OUT", (out_pt[0] - 20, out_pt[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Draw buffer zone (semi-transparent)
    buf = state.buffer_px
    for sign in [1, -1]:
        bpt1 = (int(pt1[0] + nx * buf * sign), int(pt1[1] + ny * buf * sign))
        bpt2 = (int(pt2[0] + nx * buf * sign), int(pt2[1] + ny * buf * sign))
        cv2.line(vis, bpt1, bpt2, (128, 128, 128), 1, cv2.LINE_AA)

    # Last click result
    if state.last_click is not None:
        cx, cy = state.last_click
        side = state.last_side
        if side == 0:
            label, color = "BUFFER", (128, 128, 128)
        elif (side == 1 and state.enter_direction == "positive") or \
             (side == -1 and state.enter_direction == "negative"):
            label, color = "IN", (0, 255, 0)
        else:
            label, color = "OUT", (0, 0, 255)

        cv2.circle(vis, (cx, cy), 10, color, -1)
        cv2.putText(vis, label, (cx + 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Help text
    h = vis.shape[0]
    help_lines = [
        f"enter_direction: {state.enter_direction}",
        f"buffer_px: {state.buffer_px}",
        "---",
        "Click: test side | F: flip IN/OUT | S: save | P: pick new line | Q: quit",
    ]
    if state.picking_mode:
        picked = len(state.picked_points)
        help_lines.append(f"PICKING MODE: click point {picked+1}/2")

    for i, line in enumerate(help_lines):
        cv2.putText(vis, line, (10, h - 20 - (len(help_lines) - 1 - i) * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------

def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state.picking_mode:
        state.picked_points.append((x, y))
        if len(state.picked_points) == 2:
            state.pt1 = state.picked_points[0]
            state.pt2 = state.picked_points[1]
            state.picking_mode = False
            state.picked_points = []
            print(f"[line] new points: {list(state.pt1)}, {list(state.pt2)}")
        return

    if state.pt1 is not None and state.pt2 is not None:
        side = point_side_of_line((x, y), state.pt1, state.pt2, state.buffer_px)
        state.last_click = (x, y)
        state.last_side = side

        if side == 0:
            label = "BUFFER"
        elif (side == 1 and state.enter_direction == "positive") or \
             (side == -1 and state.enter_direction == "negative"):
            label = "IN"
        else:
            label = "OUT"
        print(f"[click] ({x}, {y}) → side={side} → {label}")


# ---------------------------------------------------------------------------
# Config save
# ---------------------------------------------------------------------------

def save_config(cfg, config_path):
    lines = cfg["counting"]["crosslines"][state.line_cfg_index]
    lines["points"] = [list(state.pt1), list(state.pt2)]
    lines["enter_direction"] = state.enter_direction
    lines["buffer_px"] = state.buffer_px

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[save] config saved to {config_path}")
    print(f"       points: {lines['points']}")
    print(f"       enter_direction: {state.enter_direction}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Crossline calibration tool")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--line", type=int, default=0, help="Crossline index (if multiple)")
    args = parser.parse_args()

    state.config_path = args.config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    source = args.source or cfg["video"]["source"]
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    crosslines = cfg.get("counting", {}).get("crosslines", [])
    if not crosslines:
        sys.exit("No crosslines defined in config. Add one under counting.crosslines first.")

    state.line_cfg_index = args.line
    line_cfg = crosslines[state.line_cfg_index]
    pt1_raw, pt2_raw = line_cfg["points"]
    state.pt1 = tuple(pt1_raw)
    state.pt2 = tuple(pt2_raw)
    state.enter_direction = line_cfg.get("enter_direction", "positive")
    state.buffer_px = float(line_cfg.get("buffer_px", 20))

    cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))
    if not cap.isOpened():
        sys.exit(f"Cannot open: {source}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("Cannot read first frame")

    state.frame = frame
    win = "Crossline Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, frame.shape[1]), min(900, frame.shape[0]))
    cv2.setMouseCallback(win, on_mouse)

    print(f"[calibrate] line: {line_cfg.get('name', 'unnamed')}")
    print(f"[calibrate] points: {list(state.pt1)} → {list(state.pt2)}")
    print(f"[calibrate] enter_direction: {state.enter_direction}")
    print(f"[calibrate] Click on frame to test IN/OUT side")
    print(f"[calibrate] F=flip, S=save, P=pick new line, Q=quit")

    while True:
        vis = draw_overlay(state.frame)
        cv2.imshow(win, vis)

        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):  # Q or ESC
            break

        elif key == ord("f"):
            # Flip direction
            state.enter_direction = "negative" if state.enter_direction == "positive" else "positive"
            state.last_click = None
            print(f"[flip] enter_direction → {state.enter_direction}")

        elif key == ord("s"):
            save_config(cfg, args.config)

        elif key == ord("p"):
            state.picking_mode = True
            state.picked_points = []
            print("[pick] Click 2 points to define new line")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
