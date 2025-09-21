from __future__ import annotations
from typing import Optional, Tuple, Any, Dict
from pathlib import Path
import cv2
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency handled gracefully
    yaml = None


def _load_tracker_config() -> Dict[str, Any]:
    """Load tracker config from tracker.yaml in the same directory as this file."""
    # Start with empty config; rely on YAML if present
    config: Dict[str, Any] = {"csrt_params": {}}

    # Use tracker.yaml in the same directory as this file
    candidate = Path(__file__).with_name("tracker.yaml")

    # Try to load YAML
    if yaml is not None and candidate.exists():
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                # Shallow merge for known sections
                if "csrt_params" in loaded and isinstance(loaded["csrt_params"], dict):
                    config["csrt_params"].update(loaded["csrt_params"])  # type: ignore[arg-type]
        except Exception:
            # Fall back silently to defaults
            pass
    return config


def make_tracker():
    """Create CSRT tracker with parameters loaded from YAML (with sensible defaults)."""
    if hasattr(cv2, 'TrackerCSRT_create'):
        try:
            # Create parameter object
            params = cv2.TrackerCSRT_Params()
            cfg = _load_tracker_config()
            csrt_params: Dict[str, Any] = cfg.get("csrt_params", {})  # type: ignore[assignment]

            # Apply parameters present in config if attribute exists on this OpenCV build
            for name, value in csrt_params.items():
                if hasattr(params, name):
                    try:
                        setattr(params, name, value)
                    except Exception:
                        # Continue on best-effort basis if a field rejects value type
                        pass

            # Create tracker with configured params
            return cv2.TrackerCSRT_create(params)
            
        except Exception as e:
            raise ImportError(
                f"Failed to create CSRT tracker: {e}\n"
                "Please install opencv-contrib-python: pip install opencv-contrib-python"
            )
    
    # CSRT tracker not available - raise error
    raise ImportError(
        "CSRT tracker is not available. "
        "Please install opencv-contrib-python: pip install opencv-contrib-python"
    )


class ObjectTracker:
    def __init__(self):
        self.tracker = None
        self.initialized = False
        

    def init(self, frame, bbox_xywh: Tuple[int,int,int,int]):
        self.tracker = make_tracker()
        x,y,w,h = map(int, bbox_xywh)
        self.initialized = self.tracker.init(frame, (x,y,w,h))

    def update(self, frame) -> Optional[Tuple[int,int,int,int]]:
        if not self.tracker:
            return None
        ok, box = self.tracker.update(frame)
        if not ok:
            return None
        x, y, w, h = box
        return (int(x), int(y), int(w), int(h))