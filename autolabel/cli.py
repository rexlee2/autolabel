from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from .ui_utils import (
    compute_initial_window_size,
    get_window_size,
    center_window,
    fit_to_window_letterbox,
    map_window_to_image,
    compute_center_drag_bbox,
    read_frame,
)

from .ui_utils import AnnotationWriter, draw_hud, draw_rect, clamp_bbox
from .tracker import ObjectTracker  # Use the enhanced trackera


@dataclass
class State:
    frame_idx: int = 0
    mode: str = "REVIEW"            # REVIEW, FIX
    bbox_xywh: Optional[Tuple[int,int,int,int]] = None
    draw_anchor: Optional[Tuple[int,int]] = None  # used as center point in center-drag
    tracker_on: bool = False
    quit_requested: bool = False  # for quit confirmation


class LabelUI:
    def __init__(self, video_path: Path, out_path: Path, prompt: Optional[str] = None):
        self.video_path = str(video_path)
        self.out_path = Path(out_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.state = State()
        self.writer = AnnotationWriter(out_path)
        self.tracker = None  # Initialize to None, create when needed
        self.prompt = prompt
        self.prompt_detector = PromptDetector() if prompt else None
        self._win = "Annotator"
        # Compute initial window size: ~80% of screen height, preserving aspect ratio
        init_w, init_h = compute_initial_window_size(self.w, self.h)
        self._initial_win_size = (init_w, init_h)

        # Create resizable window and set initial size (~80% vertical)
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        try:
            cv2.resizeWindow(self._win, init_w, init_h)
        except Exception:
            pass
        cv2.setMouseCallback(self._win, self._on_mouse)
        self._hud_h = 0
        self._track_conf_threshold = 0.35
        self._window_centered = False
        # Track how the image is mapped into the window for correct mouse mapping
        self._disp_scale = 1.0
        self._disp_pad_x = 0
        self._disp_pad_y = 0

    # ---------------- MOUSE ----------------
    def _on_mouse(self, event, x, y, flags, param):
        if self.state.mode not in ("FIX",):
            return
        mapped = map_window_to_image(
            x_win=x,
            y_win=y,
            display_scale=self._disp_scale,
            pad_x=self._disp_pad_x,
            pad_y=self._disp_pad_y,
            hud_height=self._hud_h,
            image_width=self.w,
            image_height=self.h,
        )
        if mapped is None:
            return
        x_img, y_img = mapped

        if event == cv2.EVENT_LBUTTONDOWN:
            # Center-based drawing: first click defines center in original image coords
            self.state.draw_anchor = (x_img, y_img)
            self.state.bbox_xywh = (max(0, x_img - 1), max(0, y_img - 1), 2, 2)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self.state.draw_anchor:
                cx, cy = self.state.draw_anchor
                x0, y0, w, h = compute_center_drag_bbox(cx, cy, x_img, y_img, self.w, self.h)
                self.state.bbox_xywh = (x0, y0, w, h)
        elif event == cv2.EVENT_LBUTTONUP:
            self.state.draw_anchor = None

    # --------------- RENDER ---------------
    def _render(self, frame, legend: str, header: str, legend_scale: float = 1.0, legend_plain: bool = False):
        img = frame.copy()
        if self.state.bbox_xywh is not None:
            draw_rect(img, self.state.bbox_xywh)
        img_with_hud, hud_h = draw_hud(img, header, legend, legend_scale=legend_scale, legend_plain=legend_plain)
        self._hud_h = hud_h
        # Scale image to current window size (follow user resizes) with aspect-ratio preserving letterbox
        try:
            win_w, win_h = get_window_size(self._win, self._initial_win_size)
            display_img, self._disp_scale, self._disp_pad_x, self._disp_pad_y = fit_to_window_letterbox(img_with_hud, win_w, win_h)
        except Exception:
            display_img = img_with_hud
        cv2.imshow(self._win, display_img)
        if not self._window_centered:
            center_window(self._win, self._initial_win_size)
            self._window_centered = True


    # --------------- FLOW -----------------
    def _read_frame(self, idx: int):
        self.cap, frame = read_frame(self.cap, self.video_path, idx, self.total)
        return frame

    def _init_tracker(self, frame, bbox):
        """Initialize tracker with first bbox."""
        self.tracker = ObjectTracker()
        self.tracker.init(frame, bbox)
        self.state.tracker_on = True

    def _track_next(self, frame):
        """Update tracker with current frame."""
        if self.tracker is None or not self.state.tracker_on:
            return None
        
        pred = self.tracker.update(frame)
        
        # If tracker fails, reset tracking state
        if pred is None:
            self.state.tracker_on = False
            
        return pred

    def _pad_remaining_as_skips(self) -> None:
        """Ensure the annotation file has one record per frame by padding
        remaining frames with Skip entries."""
        try:
            while self.writer.count < self.total:
                self.writer.write_skip()
        except Exception:
            # If anything goes wrong, fail safe without crashing the UI
            pass

    def _flush_pending_current_bbox(self) -> None:
        """If the current frame has an unsaved bbox, write it as Visible.
        Consider a bbox 'pending' when a rectangle exists for the current
        frame index and no annotation has yet been written for this frame.
        """
        try:
            if self.state.bbox_xywh is not None and self.writer.count == self.state.frame_idx:
                bbox = clamp_bbox(self.state.bbox_xywh, self.w, self.h)
                self.writer.write_visible(bbox)
        except Exception:
            pass

    def run(self):
        # Main loop across frames
        aborted = False  # Track if user aborted the annotation
        terminated = False  # True if user chose to finish (save or exit) via overlay
        while self.state.frame_idx < self.total:
            frame = self._read_frame(self.state.frame_idx)
            if frame is None:
                break

            header = f"Frame {self.state.frame_idx+1}/{self.total} | Mode: {self.state.mode}"

            # Mode handlers (only REVIEW and FIX)
            if self.state.mode == "REVIEW":
                has_label = (self.state.frame_idx < self.writer.count) or (self.state.bbox_xywh is not None)
                display_mode = "REVIEW" if has_label else "LABEL"
                header2 = f"Frame {self.state.frame_idx+1}/{self.total} | Mode: {display_mode}"
                if has_label:
                    legend = "A=Accept  F=Fix  S=Skip  I=Invisible  Q=Quit"
                else:
                    legend = "L=Label  S=Skip  I=Invisible  Q=Quit"
                self._render(frame, legend, header2)
                key = cv2.waitKey(20)
                key8 = key & 0xFF

                if key8 in (ord('f'), ord('F')):
                    # Only allow Fix if there is a label to fix
                    if has_label:
                        self.state.mode = "FIX"
                    self.state.quit_requested = False
                elif key8 in (ord('l'), ord('L')):
                    pred = None
                    if self.prompt_detector is not None:
                        pred = self.prompt_detector.detect(frame, self.prompt)  # type: ignore[arg-type]
                        if pred is not None:
                            self.state.bbox_xywh = clamp_bbox(pred, self.w, self.h)
                            # Reinitialize tracker with new detection
                            self._init_tracker(frame, self.state.bbox_xywh)
                            self.state.quit_requested = False
                            continue
                    # Fallback to manual drawing in FIX mode
                    self.state.bbox_xywh = None
                    self.state.mode = "FIX"
                    self.state.quit_requested = False
                elif key8 in (ord('a'), ord('A')):  # Accept only via 'A' in REVIEW
                    if has_label:
                        if self.writer.count > self.state.frame_idx:
                            # Already saved for this frame; just advance and try to track next if applicable
                            self.state.frame_idx += 1
                            if self.state.frame_idx >= self.total:
                                break
                            next_frame = self._read_frame(self.state.frame_idx)
                            if next_frame is not None and self.state.tracker_on:
                                pred = self._track_next(next_frame)
                                self.state.bbox_xywh = pred
                            else:
                                self.state.bbox_xywh = None
                            self.state.mode = "REVIEW"
                            self.state.quit_requested = False
                        elif self.state.bbox_xywh is not None:
                            # Save current bbox and continue tracking
                            bbox = clamp_bbox(self.state.bbox_xywh, self.w, self.h)
                            self.writer.write_visible(bbox)
                            self._init_tracker(frame, bbox)
                            self.state.frame_idx += 1
                            if self.state.frame_idx >= self.total:
                                break
                            next_frame = self._read_frame(self.state.frame_idx)
                            if next_frame is not None:
                                pred = self._track_next(next_frame)
                                self.state.bbox_xywh = pred
                            else:
                                self.state.bbox_xywh = None
                            self.state.mode = "REVIEW"
                            self.state.quit_requested = False
                elif key8 in (ord('s'), ord('S')):
                    self.writer.write_skip()
                    self.state.frame_idx += 1
                    self.state.bbox_xywh = None
                    self.state.mode = "REVIEW"
                    # Reset tracker on skip
                    self.state.tracker_on = False
                    self.tracker = None
                    self.state.quit_requested = False
                elif key8 in (ord('i'), ord('I')):
                    self.writer.write_invisible()
                    self.state.frame_idx += 1
                    self.state.bbox_xywh = None
                    self.state.mode = "REVIEW"
                    # Reset tracker on invisible
                    self.state.tracker_on = False
                    self.tracker = None
                    self.state.quit_requested = False
                elif key8 in (ord('q'), ord('Q')):
                    # Show save/exit overlay immediately
                    try:
                        last_frame = frame

                        header2 = "Save annotations? ENTER=Save  Q=Exit without saving  ESC=Escape"
                        path_str = str(self.out_path.resolve())
                        legend2 = f"Save to: {path_str}"
                        self._render(last_frame, legend2, header2, legend_scale=0.7, legend_plain=True)

                        action = None
                        while True:
                            k2 = cv2.waitKey(0)
                            k2_8 = k2 & 0xFF
                            if k2_8 in (13, 10):
                                self._flush_pending_current_bbox()
                                self._pad_remaining_as_skips()
                                self.writer.close()
                                print(f"Saved annotations to {self.out_path}")
                                action = 'save'
                                break
                            elif k2_8 in (ord('q'), ord('Q')):
                                print("Exiting without saving.")
                                action = 'exit'
                                break
                            elif k2_8 == 27:  # ESC
                                action = 'cancel'
                                break
                    except Exception:
                        action = 'exit'
                    if action in ('save', 'exit'):
                        terminated = True
                        break
                    else:
                        self.state.quit_requested = False
                        continue
                else:
                    if key8 != 255:  # Any other key resets quit request
                        self.state.quit_requested = False
                    continue

            elif self.state.mode == "FIX":
                legend = "Draw box with mouse. ENTER=Confirm  ESC=Cancel"
                self._render(frame, legend, header)
                key = cv2.waitKey(20)
                key8 = key & 0xFF

                if key8 in (13, 10):  # Enter 
                    if self.state.bbox_xywh is not None:
                        self.state.bbox_xywh = clamp_bbox(self.state.bbox_xywh, self.w, self.h)
                        self.writer.write_visible(self.state.bbox_xywh)
                        
                        # Reinitialize tracker with corrected bbox
                        self._init_tracker(frame, self.state.bbox_xywh)
                        
                        # Move to next frame
                        self.state.frame_idx += 1
                        if self.state.frame_idx >= self.total:
                            break
                        
                        # Continue tracking
                        next_frame = self._read_frame(self.state.frame_idx)
                        if next_frame is not None:
                            pred = self._track_next(next_frame)
                            self.state.bbox_xywh = pred
                        else:
                            self.state.bbox_xywh = None
                        
                        self.state.mode = "REVIEW"
                    self.state.quit_requested = False
                elif key8 == 27:  # ESC cancel
                    self.state.mode = "REVIEW"
                    self.state.quit_requested = False
                elif key8 in (ord('q'), ord('Q')):
                    # Show save/exit overlay immediately
                    try:
                        last_frame = frame

                        header2 = "Save annotations? ENTER=Save  Q=Exit without saving  ESC=Escape"
                        path_str = str(self.out_path.resolve())
                        legend2 = f"Will save to: {path_str}"
                        self._render(last_frame, legend2, header2, legend_scale=0.7, legend_plain=True)

                        action = None
                        while True:
                            k2 = cv2.waitKey(0)
                            k2_8 = k2 & 0xFF
                            if k2_8 in (13, 10):
                                self._flush_pending_current_bbox()
                                self._pad_remaining_as_skips()
                                self.writer.close()
                                print(f"Saved annotations to {self.out_path}")
                                action = 'save'
                                break
                            elif k2_8 in (ord('q'), ord('Q')):
                                print("Exiting without saving.")
                                action = 'exit'
                                break
                            elif k2_8 == 27:  # ESC
                                action = 'cancel'
                                break
                    except Exception:
                        action = 'exit'
                    if action in ('save', 'exit'):
                        terminated = True
                        break
                    else:
                        self.state.quit_requested = False
                        continue
                else:
                    if key8 != 255:  # Any other key resets quit request
                        self.state.quit_requested = False
                    continue

        # Final completion notice before closing  
        if not terminated and not aborted:
            try:
                # Load the last available frame on demand
                last_frame = self._read_frame(max(0, min(self.total - 1, self.state.frame_idx)))
                if last_frame is None:
                    last_frame = np.zeros((max(100, self.h), max(200, self.w), 3), dtype=np.uint8)

                header = "Annotation complete! ENTER=Save  Q=Abandon"
                path_str = str(self.out_path.resolve())
                legend = f"Save to: {path_str}"
                self._render(last_frame, legend, header, legend_scale=0.7, legend_plain=True)

                # Wait for Enter (save) or Q (abandon)
                while True:
                    key = cv2.waitKey(0)
                    key8 = key & 0xFF
                    if key8 in (13, 10):  # Enter - save
                        self._flush_pending_current_bbox()
                        self._pad_remaining_as_skips()
                        self.writer.close()
                        print(f"Saved annotations to {self.out_path}")
                        break
                    elif key8 in (ord('q'), ord('Q')):  # Q - abandon
                        print("Annotation abandoned. File not saved.")
                        # Delete the file if it was created
                        try:
                            if self.out_path.exists():
                                self.out_path.unlink()
                        except Exception:
                            pass
                        break
            except Exception:
                pass
        elif not terminated:
            # User aborted mid-sequence
            print("Annotation aborted. File not saved.")
            try:
                if self.out_path.exists():
                    self.out_path.unlink()
            except Exception:
                pass

        self.cap.release()
        cv2.destroyAllWindows()


def run(video_path: Path, out_path: Path, prompt: Optional[str] = None):
    ui = LabelUI(video_path, out_path, prompt=prompt)
    ui.run()


def main():
    parser = argparse.ArgumentParser(description="Tracking Annotator (Accept/Fix)")
    parser.add_argument(
        "video",
        type=Path,
        nargs='?',
        default=Path("assets/sample_video.mp4"),
        help="Path to input video file (defaults to assets/sample_video.mp4)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output annotation file path (defaults to <video>.annotations)")
    parser.add_argument("--prompt", type=str, default=None, help="Optional text prompt for prompt-based initial detection")
    args = parser.parse_args()

    video: Path = args.video
    out_path: Path = args.out if args.out is not None else video.with_suffix(".annotations")
    run(video, out_path, prompt=args.prompt)


if __name__ == "__main__":
    main()