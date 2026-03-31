"""
Công cụ chọn tọa độ chính xác cho line/zone từ video hoặc ảnh.

Cách dùng:
    python tools/pick_coordinates.py --source video.mp4
    python tools/pick_coordinates.py --source video.mp4 --frame 100
    python tools/pick_coordinates.py --source snapshot.jpg

Phím tắt:
    Click trái  — thêm điểm
    r           — xoá điểm cuối
    c           — xoá tất cả
    s           — in tọa độ ra terminal (copy vào config.yaml)
    → / ←       — next/prev frame (khi dùng video)
    q           — thoát
"""

import argparse
import sys
import cv2
import numpy as np

# Kích thước cửa sổ hiển thị tối đa (không thay đổi ảnh gốc)
MAX_W = 1280
MAX_H = 720

# Zoom box: kích thước vùng ảnh gốc xung quanh cursor sẽ được phóng to
ZOOM_REGION = 80   # px xung quanh cursor (trên ảnh gốc)
ZOOM_SIZE   = 200  # kích thước của ô zoom hiển thị (px)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
points_orig = []      # tọa độ thực trên ảnh gốc
cursor_orig = (0, 0)  # vị trí cursor trên ảnh gốc (realtime)
frame_orig  = None    # ảnh gốc (không bao giờ bị vẽ lên)
scale       = 1.0     # display_px = orig_px * scale


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def compute_scale(orig_w: int, orig_h: int) -> float:
    """Tính scale để ảnh vừa trong MAX_W × MAX_H."""
    return min(MAX_W / orig_w, MAX_H / orig_h, 1.0)


def disp_to_orig(dx: int, dy: int) -> tuple:
    """Chuyển tọa độ display → tọa độ ảnh gốc."""
    return (round(dx / scale), round(dy / scale))


def orig_to_disp(ox: int, oy: int) -> tuple:
    """Chuyển tọa độ ảnh gốc → tọa độ display."""
    return (round(ox * scale), round(oy * scale))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(frame_orig: np.ndarray) -> np.ndarray:
    """Tạo frame display từ ảnh gốc + tất cả annotations."""
    h, w = frame_orig.shape[:2]
    dw, dh = round(w * scale), round(h * scale)
    disp = cv2.resize(frame_orig, (dw, dh), interpolation=cv2.INTER_LINEAR)

    # --- Vẽ các điểm và đường ---
    disp_pts = [orig_to_disp(ox, oy) for ox, oy in points_orig]

    if len(disp_pts) >= 2:
        for i in range(1, len(disp_pts)):
            cv2.line(disp, disp_pts[i - 1], disp_pts[i], (0, 255, 0), 2)
    if len(disp_pts) >= 3:
        cv2.polylines(disp, [np.array(disp_pts, np.int32)],
                      isClosed=True, color=(0, 165, 255), thickness=2)

    for i, (dx, dy) in enumerate(disp_pts):
        cv2.circle(disp, (dx, dy), 6, (0, 0, 255), -1)
        cv2.putText(disp, str(i + 1), (dx + 8, dy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # --- Crosshair tại cursor ---
    cx, cy = orig_to_disp(*cursor_orig)
    cv2.line(disp, (cx, 0), (cx, dh), (255, 255, 0), 1)
    cv2.line(disp, (0, cy), (dw, cy), (255, 255, 0), 1)

    # --- Tọa độ gốc realtime ---
    ox, oy = cursor_orig
    coord_text = f"({ox}, {oy})"
    cv2.putText(disp, coord_text, (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    # --- Zoom box góc trên phải ---
    disp = draw_zoom_box(disp, frame_orig, ox, oy)

    # --- Hướng dẫn ---
    guide = "Click=diem | r=xoa | c=xoa het | s=luu | q=thoat"
    cv2.putText(disp, guide, (8, dh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # --- Danh sách điểm đã chọn ---
    for i, (ox2, oy2) in enumerate(points_orig):
        cv2.putText(disp, f"P{i+1}: ({ox2},{oy2})", (8, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    return disp


def draw_zoom_box(disp: np.ndarray, orig: np.ndarray,
                  cx: int, cy: int) -> np.ndarray:
    """Vẽ ô zoom phóng to vùng xung quanh cursor lên góc trên phải của disp."""
    oh, ow = orig.shape[:2]
    r = ZOOM_REGION

    x1 = max(cx - r, 0);  x2 = min(cx + r, ow)
    y1 = max(cy - r, 0);  y2 = min(cy + r, oh)

    crop = orig[y1:y2, x1:x2]
    if crop.size == 0:
        return disp

    zoom = cv2.resize(crop, (ZOOM_SIZE, ZOOM_SIZE), interpolation=cv2.INTER_LINEAR)

    # Vẽ crosshair chính giữa ô zoom
    zc = ZOOM_SIZE // 2
    off_x = round((cx - x1) / (x2 - x1) * ZOOM_SIZE) if x2 > x1 else zc
    off_y = round((cy - y1) / (y2 - y1) * ZOOM_SIZE) if y2 > y1 else zc
    cv2.line(zoom, (off_x, 0), (off_x, ZOOM_SIZE), (255, 255, 0), 1)
    cv2.line(zoom, (0, off_y), (ZOOM_SIZE, off_y), (255, 255, 0), 1)
    cv2.circle(zoom, (off_x, off_y), 4, (0, 0, 255), -1)

    # Dán vào góc trên phải
    dh, dw = disp.shape[:2]
    tx = dw - ZOOM_SIZE - 10
    ty = 10
    disp[ty:ty + ZOOM_SIZE, tx:tx + ZOOM_SIZE] = zoom
    cv2.rectangle(disp, (tx, ty), (tx + ZOOM_SIZE, ty + ZOOM_SIZE),
                  (255, 255, 255), 2)
    cv2.putText(disp, "ZOOM", (tx + 4, ty + ZOOM_SIZE + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return disp


# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------

def on_mouse(event, x, y, flags, param):
    global cursor_orig
    cursor_orig = disp_to_orig(x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        points_orig.append(cursor_orig)

    # Redraw mỗi khi chuột di chuyển hoặc click
    if frame_orig is not None:
        cv2.imshow("Pick Coordinates", render(frame_orig))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_result():
    n = len(points_orig)
    if n == 0:
        print("Chua chon diem nao.")
        return

    print("\n" + "=" * 50)
    if n == 2:
        print("--- CROSSLINE (dan vao config.yaml) ---")
        print(f"      points: [{list(points_orig[0])}, {list(points_orig[1])}]")
    elif n >= 3:
        pts_str = ", ".join(str(list(p)) for p in points_orig)
        print("--- ZONE (dan vao config.yaml) ---")
        print(f"      points: [{pts_str}]")
    else:
        print(f"Diem da chon: {[list(p) for p in points_orig]}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_frame(source: str, frame_idx: int):
    """Trả về (frame_bgr, cap_or_None)."""
    ext = source.lower().rsplit(".", 1)[-1]
    if ext in ("jpg", "jpeg", "png", "bmp", "tiff"):
        img = cv2.imread(source)
        if img is None:
            sys.exit(f"Khong doc duoc anh: {source!r}")
        return img, None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Khong mo duoc video: {source!r}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        sys.exit(f"Khong doc duoc frame {frame_idx}")
    return frame, cap


def main():
    global frame_orig, scale

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame thu N de lay lam nen (mac dinh: 0)")
    args = parser.parse_args()

    frame_orig, cap = load_frame(args.source, args.frame)
    oh, ow = frame_orig.shape[:2]
    scale = compute_scale(ow, oh)
    current_frame = args.frame

    print(f"Anh goc: {ow} x {oh}  |  Scale hien thi: {scale:.3f}")
    print(f"Click de chon toa do. Nhan 's' de in ket qua.")

    cv2.namedWindow("Pick Coordinates", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Pick Coordinates", on_mouse)
    cv2.imshow("Pick Coordinates", render(frame_orig))

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            if points_orig:
                points_orig.pop()
                cv2.imshow("Pick Coordinates", render(frame_orig))

        elif key == ord("c"):
            points_orig.clear()
            cv2.imshow("Pick Coordinates", render(frame_orig))

        elif key == ord("s"):
            print_result()

        # Di chuyển frame (chỉ khi là video)
        elif cap is not None and key in (82, 83, 2555904, 2424832,  # arrow keys
                                          ord("."), ord(",")):
            step = 10 if key in (ord("."), 83, 2555904) else -10
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = max(0, min(current_frame + step, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame_orig = cap.read()
            if ret:
                cv2.imshow("Pick Coordinates", render(frame_orig))

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
