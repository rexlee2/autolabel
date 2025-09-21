"""
AutoLabel - Video object annotation tool with tracking
"""
__version__ = "1.0.0"

from .tracker import ObjectTracker
from .ui_utils import (
    AnnotationWriter,
    draw_hud,
    draw_rect,
    clamp_bbox,
    compute_initial_window_size,
    get_window_size,
    center_window,
    fit_to_window_letterbox,
    map_window_to_image,
    compute_center_drag_bbox,
    bbox_center_to_xywh,
    bbox_xywh_to_center,
    read_frame,
)

__all__ = [
    "ObjectTracker",
    "PromptDetector",
    "AnnotationWriter",
    "draw_hud",
    "draw_rect",
    "clamp_bbox",
    "compute_initial_window_size",
    "get_window_size",
    "center_window",
    "fit_to_window_letterbox",
    "map_window_to_image",
    "compute_center_drag_bbox",
    "bbox_center_to_xywh",
    "bbox_xywh_to_center",
    "read_frame",
]