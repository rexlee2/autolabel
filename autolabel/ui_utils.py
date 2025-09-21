from __future__ import annotations
from typing import Tuple, Optional, Union
from pathlib import Path
import ctypes
import cv2
import numpy as np


def _set_process_dpi_aware() -> None:
    """Best-effort DPI awareness to get correct screen metrics on Windows."""
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _get_screen_size() -> Tuple[int, int]:
    """Return (screen_w, screen_h); falls back to (0, 0) if unavailable."""
    try:
        _set_process_dpi_aware()
        screen_w = int(ctypes.windll.user32.GetSystemMetrics(0))
        screen_h = int(ctypes.windll.user32.GetSystemMetrics(1))
        return screen_w, screen_h
    except Exception:
        return 0, 0


def compute_initial_window_size(
    image_w: int,
    image_h: int,
    vertical_ratio: float = 0.8,
    min_w: int = 320,
    min_h: int = 240,
    screen_margin_ratio: float = 0.95,
) -> Tuple[int, int]:
    """Compute an initial window size preserving aspect ratio and using ~80% of screen height.
    Returns (init_w, init_h).
    """
    try:
        screen_w, screen_h = _get_screen_size()
        target_h = int(max(200, round((screen_h or 1080) * vertical_ratio)))
        target_w = int(max(300, round(target_h * (image_w / max(1, image_h)))))
        if screen_w and target_w > int(screen_w * screen_margin_ratio):
            target_w = int(screen_w * screen_margin_ratio)
            target_h = int(target_w * (image_h / max(1, image_w)))
    except Exception:
        target_h = 864
        target_w = int(target_h * (image_w / max(1, image_h)))

    init_w = max(min_w, int(target_w))
    init_h = max(min_h, int(target_h))
    return init_w, init_h


def get_window_size(window_name: str, default_size: Tuple[int, int]) -> Tuple[int, int]:
    """Get current window width/height or fall back to provided default."""
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
        return int(win_w), int(win_h)
    except Exception:
        return int(default_size[0]), int(default_size[1])


def center_window(window_name: str, default_size: Tuple[int, int]) -> None:
    """Center a window on the primary screen using current size or default_size."""
    try:
        screen_w, screen_h = _get_screen_size()
        win_w, win_h = get_window_size(window_name, default_size)
        x = max(0, (int(screen_w) - int(win_w)) // 2) if screen_w else 0
        y = max(0, (int(screen_h) - int(win_h)) // 2) if screen_h else 0
        cv2.moveWindow(window_name, x, y)
    except Exception:
        pass


def fit_to_window_letterbox(
    img: np.ndarray,
    target_w: int,
    target_h: int,
) -> Tuple[np.ndarray, float, int, int]:
    """Resize img to fit target window (letterbox), returning (canvas, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    if w <= 0 or h <= 0 or target_w <= 0 or target_h <= 0:
        return img, 1.0, 0, 0
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    if new_w == target_w and new_h == target_h:
        return resized, scale, 0, 0
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, scale, x0, y0


def map_window_to_image(
    x_win: int,
    y_win: int,
    display_scale: float,
    pad_x: int,
    pad_y: int,
    hud_height: int,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, int]]:
    """Map window coordinates to image coordinates given letterbox transform and HUD height.
    Returns (x_img, y_img) as ints clamped to image bounds, or None if outside content/HUD.
    """
    try:
        scale = display_scale if display_scale and display_scale > 0 else 1.0
        if x_win < pad_x or y_win < pad_y:
            return None
        x_img_f = (x_win - pad_x) / scale
        y_img_total_f = (y_win - pad_y) / scale
        if y_img_total_f < hud_height:
            return None
        y_img_f = y_img_total_f - hud_height
        # Ignore clicks below the content (e.g., in bottom border)
        if y_img_f >= image_height:
            return None
        x_img = max(0, min(image_width - 1, int(round(x_img_f))))
        y_img = max(0, min(image_height - 1, int(round(y_img_f))))
        return x_img, y_img
    except Exception:
        return None


def compute_center_drag_bbox(
    center_x: int,
    center_y: int,
    current_x: int,
    current_y: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Given a center anchor and current point, compute a clamped bbox (x, y, w, h)."""
    dx = abs(current_x - center_x)
    dy = abs(current_y - center_y)
    width = max(2, 2 * int(round(dx)))
    height = max(2, 2 * int(round(dy)))
    x0 = int(round(center_x - width / 2))
    y0 = int(round(center_y - height / 2))
    x0 = max(0, min(image_width - 2, x0))
    y0 = max(0, min(image_height - 2, y0))
    if x0 + width > image_width:
        width = image_width - x0
    if y0 + height > image_height:
        height = image_height - y0
    return x0, y0, int(width), int(height)


# -----------------------------
# Geometry utils 
# -----------------------------
def bbox_center_to_xywh(cx: int, cy: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x = int(round(cx - w / 2))
    y = int(round(cy - h / 2))
    return (x, y, int(w), int(h))


def bbox_xywh_to_center(xywh: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    x, y, w, h = map(int, xywh)
    cx = int(round(x + w / 2))
    cy = int(round(y + h / 2))
    return (cx, cy, int(w), int(h))


def clamp_bbox(xywh: Tuple[int,int,int,int], w_img: int, h_img: int) -> Tuple[int,int,int,int]:
    x, y, w, h = xywh
    x = max(0, min(x, w_img-1))
    y = max(0, min(y, h_img-1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    return (int(x), int(y), int(w), int(h))


# -----------------------------
# HUD / drawing 
# -----------------------------
YELLOW = (0, 255, 255)


def draw_hud(img: np.ndarray, text1: str, legend: str, legend_scale: float = 1.0, legend_plain: bool = False) -> Tuple[np.ndarray, int]:
    """Return a new image with a HUD bar placed ABOVE the frame.

    The original frame content is not covered; instead a top bar is added
    with dynamically scaled, outlined text for readability.
    """
    h_img, w_img = img.shape[:2]

    base_scale = max(0.65, min(1.4, h_img / 720.0))
    s1 = base_scale * 0.85
    s2 = base_scale * 0.7 * max(0.3, min(1.5, legend_scale))
    t = max(1, int(round(base_scale * 1.6)))
    pad = int(round(8 * base_scale))
    gap = int(round(12 * base_scale))

    font_header = cv2.FONT_HERSHEY_SIMPLEX
    font_legend = cv2.FONT_HERSHEY_SIMPLEX
    (w1, h1), _ = cv2.getTextSize(text1, font_header, s1, t)
    (w2, h2), b2 = cv2.getTextSize(legend, font_legend, s2, t)

    bar_h = h1 + h2 + b2 + 2 * pad + gap
    bar = np.zeros((bar_h, w_img, 3), dtype=img.dtype)

    text_color = (255, 255, 255)
    outline_color = (0, 0, 0)
    outline_t = t + 2

    p1 = (10 + pad, pad + h1)
    cv2.putText(bar, text1, p1, font_header, s1, outline_color, outline_t, cv2.LINE_AA)
    cv2.putText(bar, text1, p1, font_header, s1, text_color, t, cv2.LINE_AA)

    row2_top = pad + h1 + gap
    row2_height = h2 + b2
    row2_center = row2_top + row2_height // 2
    baseline_y = int(round(row2_center + (h2 / 2) - (b2 / 2)))
    p2 = (10 + pad, baseline_y)
    if legend:
        if legend_plain:
            cv2.putText(bar, legend, p2, font_legend, s2, (220, 220, 220), max(1, t - 1), cv2.LINE_AA)
        else:
            cv2.putText(bar, legend, p2, font_legend, s2, outline_color, outline_t, cv2.LINE_AA)
            cv2.putText(bar, legend, p2, font_legend, s2, text_color, t, cv2.LINE_AA)

    # Add a small bottom border (textless HUD) to avoid the image appearing clipped
    bottom_h = max(4, int(round(16 * base_scale)))
    bottom_bar = np.zeros((bottom_h, w_img, 3), dtype=img.dtype)
    combined = np.vstack([bar, img, bottom_bar])
    # Return only the top HUD height for interaction logic
    return combined, bar_h


def draw_rect(img: np.ndarray, xywh: Tuple[int,int,int,int], color=YELLOW, thick:int=2) -> None:
    x,y,w,h = map(int, xywh)
    cv2.rectangle(img, (x,y), (x+w, y+h), color, thick, cv2.LINE_AA)


# -----------------------------
# Annotation file writer (migrated from annotate.py)
# -----------------------------
class AnnotationWriter:
    """Writes per-frame annotations in the assignment format:
    - Visible:   V x_center y_center width height
    - Skipped:   S -1 -1 -1 -1
    - Invisible: I -1 -1 -1 -1
    One line per frame, in order.
    """
    def __init__(self, out_path: Path):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._lines: list[str] = []
        self.count = 0

    def write_visible(self, xywh: Tuple[int,int,int,int]) -> None:
        cx, cy, w, h = bbox_xywh_to_center(xywh)
        self._lines.append(f"V {cx} {cy} {w} {h}\n")
        self.count += 1

    def write_skip(self) -> None:
        self._lines.append("S -1 -1 -1 -1\n")
        self.count += 1

    def write_invisible(self) -> None:
        self._lines.append("I -1 -1 -1 -1\n")
        self.count += 1

    def close(self) -> None:
        try:
            with open(self.out_path, 'w', encoding='utf-8') as f:
                f.writelines(self._lines)
        except Exception:
            pass


# -----------------------------
# Frame reading 
# -----------------------------
def read_frame(
    cap: cv2.VideoCapture,
    video_path: Union[str, Path],
    idx: int,
    total_frames: int,
) -> Tuple[cv2.VideoCapture, Optional[np.ndarray]]:
    """Read frame at index idx, avoiding random seeks that can hang some backends.

    Strategy:
    - If idx > current position, step forward by reading frames sequentially
    - If idx < current position, reopen and fast-forward from start
    - Allow backward seeks even when current position is at/past the end
    Returns possibly-updated cap and frame (or None on failure/out-of-range).
    """
    # Guard against out-of-range requests
    if idx < 0 or (total_frames > 0 and idx >= total_frames):
        return cap, None

    try:
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    except Exception:
        current_pos = 0

    # Forward-only stepping
    if idx > current_pos:
        steps = idx - current_pos
        for _ in range(steps):
            ok, _ = cap.read()
            if not ok:
                return cap, None
    elif idx < current_pos:
        # Reopen and fast-forward when seeking backward
        try:
            cap.release()
        except Exception:
            pass
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return cap, None
        for _ in range(idx):
            ok, _ = cap.read()
            if not ok:
                return cap, None

    ok, frame = cap.read()
    if not ok:
        return cap, None
    return cap, frame


