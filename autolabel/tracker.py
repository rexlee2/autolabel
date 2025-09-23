from __future__ import annotations
from typing import Optional, Tuple, Any, Dict
from pathlib import Path
import cv2
import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency handled gracefully
    yaml = None


# -----------------------------
# Config loading (tracker.yaml)
# -----------------------------
def _load_tracker_config() -> Dict[str, Any]:
    """
    Load tracker config from tracker.yaml in the same directory as this file.
    Supported keys (all optional):
      csrt_params: dict of cv2.TrackerCSRT_Params attributes to override
      grabcut:
        enable: bool (default True)
        enlarge_ratio: float (default 0.15)
        margin_ratio: float (default 0.02)
        iters: int (default 5)
    """
    config: Dict[str, Any] = {"csrt_params": {}}
    cfg_path = Path(__file__).with_name("tracker.yaml")
    if yaml is None or not cfg_path.exists():
        return config

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            if "csrt_params" in loaded and isinstance(loaded["csrt_params"], dict):
                config["csrt_params"] = loaded["csrt_params"]
            if "grabcut" in loaded and isinstance(loaded["grabcut"], dict):
                config["grabcut"] = loaded["grabcut"]
    except Exception:
        # Best-effort: ignore malformed yaml
        pass
    return config


# -----------------------------
# CSRT tracker factory
# -----------------------------
def make_tracker() -> Any:
    """
    Create a CSRT tracker with optional parameter overrides from tracker.yaml.
    Handles both modern and legacy OpenCV namespaces.
    """
    # Try to construct params if available
    params = None
    try:
        params = cv2.TrackerCSRT_Params()  # type: ignore[attr-defined]
    except Exception:
        try:
            params = cv2.legacy.TrackerCSRT_Params()  # type: ignore[attr-defined]
        except Exception:
            params = None

    # Apply YAML-configured params if possible
    cfg = _load_tracker_config()
    csrt_params: Dict[str, Any] = cfg.get("csrt_params", {}) 
    if params is not None and isinstance(csrt_params, dict):
        for name, value in csrt_params.items():
            if hasattr(params, name):
                try:
                    setattr(params, name, value)
                except Exception:
                    # Be tolerant of type mismatches across OpenCV builds
                    pass

    # Create tracker instance
    tracker = None
    # Preferred modern API with params
    if hasattr(cv2, "TrackerCSRT_create"):
        try:
            tracker = cv2.TrackerCSRT_create(params) if params is not None else cv2.TrackerCSRT_create()  # type: ignore[call-arg]
        except Exception:
            tracker = cv2.TrackerCSRT_create()
    # Legacy namespace
    if tracker is None and hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        try:
            tracker = cv2.legacy.TrackerCSRT_create(params) if params is not None else cv2.legacy.TrackerCSRT_create()  # type: ignore[call-arg]
        except Exception:
            tracker = cv2.legacy.TrackerCSRT_create() 
    if tracker is None:
        raise RuntimeError("CSRT tracker is not available in this OpenCV build.")
    return tracker


# -----------------------------
# Geometry helpers
# -----------------------------
def _clamp_bbox(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Clamp bbox to image bounds and ensure minimum size 1x1."""
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    w = max(1, min(int(w), img_w - x))
    h = max(1, min(int(h), img_h - y))
    return x, y, w, h


def _expand_bbox(x: int, y: int, w: int, h: int, scale: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Expand bbox by `scale` (e.g., 0.15 = +15%) about its center and clamp.
    """
    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = int(round(w * (1.0 + scale)))
    new_h = int(round(h * (1.0 + scale)))
    new_x = int(round(cx - new_w / 2.0))
    new_y = int(round(cy - new_h / 2.0))
    return _clamp_bbox(new_x, new_y, new_w, new_h, img_w, img_h)


# -----------------------------
# GrabCut refinement
# -----------------------------
def refine_bbox_with_grabcut(
    frame: np.ndarray,
    bbox_xywh: Tuple[int, int, int, int],
    enlarge_ratio: float = 0.15,  # +15% around center (per spec)
    margin_ratio: float = 0.02,   # 2% of FULL image height (per spec)
    gc_iters: int = 5,
) -> Tuple[int, int, int, int]:
    """
    1) Expand the input bbox by `enlarge_ratio` around its center.
    2) Run cv2.grabCut() initialized with the expanded rect.
    3) Compute tight FG bounding box inside the ROI (definite+probable FG).
    4) Add margin of `margin_ratio * image_height` on all sides and clamp.
    Returns (x, y, w, h) in image coordinates.
    """
    h_img, w_img = frame.shape[:2]
    x, y, w, h = map(int, bbox_xywh)
    if w <= 0 or h <= 0:
        return _clamp_bbox(x, y, w, h, w_img, h_img)

    # 1) ROI expanded by +15% around center
    rx, ry, rw, rh = _expand_bbox(x, y, w, h, enlarge_ratio, w_img, h_img)

    # 2) GrabCut with rectangle initialization
    mask = np.zeros((h_img, w_img), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (rx, ry, rw, rh)
    try:
        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, gc_iters, cv2.GC_INIT_WITH_RECT)
    except Exception:
        # If GrabCut fails (rare), return the clamped original bbox
        return _clamp_bbox(x, y, w, h, w_img, h_img)

    # Foreground = definite or probable
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    fg = fg.astype(np.uint8)

    # Constrain to ROI for stability if multiple objects present
    roi_mask = np.zeros_like(fg, dtype=np.uint8)
    roi_mask[ry:ry+rh, rx:rx+rw] = 1
    fg &= roi_mask

    ys, xs = np.where(fg > 0)
    if xs.size == 0 or ys.size == 0:
        # No FG found—fall back
        return _clamp_bbox(x, y, w, h, w_img, h_img)

    # Tight bbox around FG
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    tight_w = max(1, max_x - min_x + 1)
    tight_h = max(1, max_y - min_y + 1)

    # 3) Add 2% of full image height as margin (both sides)
    pad = int(round(margin_ratio * h_img))
    out_x = min_x - pad
    out_y = min_y - pad
    out_w = tight_w + 2 * pad
    out_h = tight_h + 2 * pad
    return _clamp_bbox(out_x, out_y, out_w, out_h, w_img, h_img)


# -----------------------------
# ObjectTracker wrapper
# -----------------------------
class ObjectTracker:
    """
    Thin wrapper around OpenCV CSRT that adds GrabCut-based box refinement.
    API:
      init(frame, bbox_xywh) -> None
      update(frame) -> Optional[Tuple[int,int,int,int]]
    """
    def __init__(self):
        self.tracker = None
        self.initialized = False
        # Cache config
        self._cfg = _load_tracker_config()

    def init(self, frame, bbox_xywh: Tuple[int, int, int, int]):
        self.tracker = make_tracker()
        x, y, w, h = map(int, bbox_xywh)
        self.initialized = self.tracker.init(frame, (x, y, w, h))  # type: ignore[attr-defined]

    def update(self, frame) -> Optional[Tuple[int, int, int, int]]:
        if not self.tracker:
            return None
        ok, box = self.tracker.update(frame)  # type: ignore[attr-defined]
        if not ok:
            return None
        x, y, w, h = map(int, box)

        # GrabCut controls (with defaults matching the spec)
        gcfg = (self._cfg.get("grabcut") if isinstance(self._cfg, dict) else None) or {}
        enable = bool(gcfg.get("enable", True))
        enlarge_ratio = float(gcfg.get("enlarge_ratio", 0.15))
        margin_ratio = float(gcfg.get("margin_ratio", 0.02))
        gc_iters = int(gcfg.get("iters", 5))

        if enable:
            return refine_bbox_with_grabcut(
                frame,
                (x, y, w, h),
                enlarge_ratio=enlarge_ratio,
                margin_ratio=margin_ratio,
                gc_iters=gc_iters,
            )
        # Fallback: no refinement
        h_img, w_img = frame.shape[:2]
        return _clamp_bbox(x, y, w, h, w_img, h_img)
