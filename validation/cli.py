from __future__ import annotations
from pathlib import Path
import cv2
import ctypes
import numpy as np
import typer
from autolabel.ui_utils import compute_initial_window_size, get_window_size, center_window, fit_to_window_letterbox, draw_hud, read_frame

YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

def parse_line(line: str):
    parts = line.strip().split()
    if not parts:
        return None
    tag = parts[0]
    if tag in ("S", "I"):
        # Support optional bbox values after S/I: "S cx cy w h"
        if len(parts) == 5:
            _, a, b, c, d = parts
            return tag, int(a), int(b), int(c), int(d)
        return tag, -1, -1, -1, -1
    if tag == "V" and len(parts) == 5:
        _, a, b, c, d = parts
        return tag, int(a), int(b), int(c), int(d)
    return None


class ValidationApp:
    def __init__(self, video_path: Path, ann_path: Path | None, headless: bool = False, out_path: Path | None = None):
        self.video_path = Path(video_path)
        self.ann_path = Path(ann_path) if ann_path is not None else self.video_path.with_suffix(".annotations")
        self.headless = headless
        # Default output path next to video if not provided
        if out_path is None:
            self.out_path = self.video_path.with_name(self.video_path.stem + "_validation.mp4")
        else:
            self.out_path = Path(out_path)

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        # Cache basic metadata
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Load annotations and preserve per-line alignment with frames.
        # Do NOT drop invalid/empty lines; keep them as None to avoid index shifts.
        with open(self.ann_path, "r", encoding="utf-8") as f:
            self.records = [parse_line(l) for l in f]
        self.frame_idx = 0
        # Track if we've centered the window (to mirror autolabel UI behavior)
        self._window_centered = False

    def _read_frame(self, idx: int):
        """Robust frame access, shared with autolabel UI."""
        self.cap, frame = read_frame(self.cap, self.video_path, idx, self.total)
        return frame

    def _render_overlay(self, frame, frame_idx: int):
        rec = self.records[frame_idx] if 0 <= frame_idx < len(self.records) else None
        overlay = frame.copy()
        if rec:
            tag, cx, cy, w, h = rec
            # Draw bbox if present for any tag
            if tag == "V" and w > 0 and h > 0:
                x = int(cx - w/2)
                y = int(cy - h/2)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), YELLOW, 2)
        return overlay

    def run(self):
        if self.headless:
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self.out_path), fourcc, fps, (width, height))
            total = 0
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                overlay = self._render_overlay(frame, total)
                # HUD (without hotkeys in headless)
                rec = self.records[total] if 0 <= total < len(self.records) else None
                status_label = ""
                if rec:
                    tag = rec[0]
                    if tag == "V":
                        status_label = "[Visible]"
                    elif tag == "S":
                        status_label = "[Skipped]"
                    elif tag == "I":
                        status_label = "[Invisible]"
                disp_idx = total + 1
                hud_text = f"Frame: {disp_idx}/{self.total}" + (f"  {status_label}" if status_label else "")
                pad = 8
                x0, y0 = 8, 8
                font = cv2.FONT_HERSHEY_DUPLEX
                scale = 0.6
                thick = 1
                size, base = cv2.getTextSize(hud_text, font, scale, thick)
                box_w = size[0] + 2 * pad
                box_h = size[1] + base + 2 * pad
                bg = overlay.copy()
                cv2.rectangle(bg, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
                alpha = 0.55
                cv2.addWeighted(bg, alpha, overlay, 1 - alpha, 0, dst=overlay)
                cv2.putText(overlay, hud_text, (x0 + pad, y0 + pad + size[1]), font, scale, WHITE, thick, cv2.LINE_AA)

                writer.write(overlay)
                total += 1
            writer.release()
            self.cap.release()
            print(f"Headless validation complete. Frames processed: {total}. Output: {self.out_path}")
            return

        # Compute initial window size: ~80% of screen height, preserving aspect ratio
        init_w, init_h = compute_initial_window_size(self.w, self.h)
        self._initial_win_size = (init_w, init_h)
        cv2.namedWindow("Validation", cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow("Validation", init_w, init_h)
        except Exception:
            pass
        last_idx = -1
        cached_frame = None
        while True:
            # Only read when the requested index changes to avoid repeated seeks
            if self.frame_idx != last_idx or cached_frame is None:
                frame = self._read_frame(self.frame_idx)
                if frame is None:
                    # At end of sequence: keep showing last cached or blank frame to allow navigation (e.g., 'P')
                    if hasattr(self, 'total') and self.total > 0 and self.frame_idx >= self.total - 1:
                        if cached_frame is None:
                            cached_frame = np.zeros((max(100, self.h), max(200, self.w), 3), dtype=np.uint8)
                        last_idx = self.frame_idx
                    else:
                        break
                else:
                    cached_frame = frame.copy()
                    last_idx = self.frame_idx
            overlay = self._render_overlay(cached_frame, self.frame_idx)

            # HUD using shared renderer for consistent style with autolabel
            rec = self.records[self.frame_idx] if 0 <= self.frame_idx < len(self.records) else None
            status_label = ""
            if rec:
                tag = rec[0]
                if tag == "V":
                    status_label = "[Visible]"
                elif tag == "S":
                    status_label = "[Skipped]"
                elif tag == "I":
                    status_label = "[Invisible]"
            disp_idx = self.frame_idx + 1
            header = f"Frame: {disp_idx}/{self.total}" + (f"  {status_label}" if status_label else "")
            legend = "n=Next  N=Skip 10  p=Previous  P=Back 10  Q=Quit"
            overlay, _ = draw_hud(overlay, header, legend, legend_scale=0.9)

            try:
                win_w, win_h = get_window_size("Validation", self._initial_win_size)
                display_img = self._fit_to_target_size(overlay, win_w, win_h)
            except Exception:
                display_img = overlay
            cv2.imshow("Validation", display_img)
            if not self._window_centered:
                center_window("Validation", self._initial_win_size)
                self._window_centered = True
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q') or k == ord('Q'):
                break
            elif k == ord('n'):
                # Single frame forward
                if hasattr(self, 'total') and self.total > 0:
                    self.frame_idx = min(self.total - 1, self.frame_idx + 1)
                else:
                    self.frame_idx += 1
            elif k == ord('N'):
                # Skip 10 frames forward
                if hasattr(self, 'total') and self.total > 0:
                    self.frame_idx = min(self.total - 1, self.frame_idx + 10)
                else:
                    self.frame_idx += 10
            elif k == ord('p'):
                # Single frame backward
                self.frame_idx = max(0, self.frame_idx - 1)
            elif k == ord('P'):
                # Skip 10 frames backward
                self.frame_idx = max(0, self.frame_idx - 10)

        self.cap.release()
        cv2.destroyAllWindows()

    def _fit_to_target_size(self, img, target_w: int, target_h: int):
        canvas, _, _, _ = fit_to_window_letterbox(img, target_w, target_h)
        return canvas


app = typer.Typer(add_help_option=True)

@app.command()
def main(
    video: Path = typer.Argument(Path("assets/sample_video.mp4"), help="Path to input video (defaults to assets/sample_video.mp4)"),
    ann: Path = typer.Option(None, "--ann", help="Path to annotation file; defaults to <video_name>.annotations"),
    headless: bool = typer.Option(False, "--headless", help="Run without UI and write an output video"),
    out: Path = typer.Option(None, "--out", help="Path to write headless preview video (defaults next to video)"),
):
    ValidationApp(video_path=video, ann_path=ann, headless=headless, out_path=out).run()


if __name__ == "__main__":
    app()