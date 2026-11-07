"""Dehazing for video (traffic cameras, construction-site cameras).

Running the still-image pipeline on every frame flickers: the airlight, the transmission map, the histogram range and
the colour balance are re-estimated each frame and move with sensor noise. `TemporalDehazer` runs the same stages in the
same order but smooths those estimates over time (exponentially). The transmission map is only smoothed where the
picture did not change between frames, so moving vehicles are never blended into the background's fog estimate.

    python dehaze_video.py foggy.mp4 clear.mp4                 # whole file, default settings
    python dehaze_video.py foggy.mp4 clear.mp4 --auto --max-dim 640 --max-frames 300

    from dehaze_video import TemporalDehazer
    d = TemporalDehazer(auto=True)
    for frame in frames:                  # RGB uint8 frames, in order
        out = d.process(frame)
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import dehaze_core as C


class TemporalDehazer:
    """Stateful frame-by-frame dehazer. `smoothing` (0..1) is the weight of the past in the temporal averages
    (0 = no smoothing = independent frames); a pixel counts as moving when it changed by more than `motion_thr` grey levels.
    `denoise` (0..1) is the weight of the previous filtered frame on static pixels: sensor noise that the sharpening and
    contrast stages would otherwise amplify and show as flicker is averaged out; moving pixels are left untouched."""

    def __init__(self, P=None, auto=False, smoothing=0.9, motion_thr=10.0, denoise=0.6):
        self.base = dict(C.P_BALANCED if P is None else P)
        self.auto, self.smoothing, self.motion_thr = auto, float(smoothing), float(motion_thr)
        self.denoise = float(denoise) if smoothing > 0 else 0.0
        self.reset()

    def reset(self):
        self.prev_gray = self.t = self.A = self.hist_range = self.gains = self.haze = self.prev_in = None
        self.P = None

    def _ema(self, old, new, w):
        return new if old is None else w * old + (1.0 - w) * new

    def process(self, frame):
        P = dict(self.base)
        w = self.smoothing
        if self.auto:                       # strength follows the (smoothed) haze level
            self.haze = self._ema(self.haze, C.haze_level(frame), w)
            g = C.strength_scale(self.haze)
            for k in ("hist", "freq", "smooth", "sharpen"):
                P[k] = P[k] * g
            if P["use_dcp"]:
                P["omega"] = min(0.95, P["omega"] * (0.35 + 0.65 * g))
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.5)
        still = (np.abs(gray - self.prev_gray) < self.motion_thr) if self.prev_gray is not None else None
        self.prev_gray = gray
        img = frame
        if self.denoise > 0 and self.prev_in is not None and self.prev_in.shape == frame.shape:
            a = np.where(still, self.denoise, 0.0).astype(np.float32)[..., None]
            img = np.clip(a * self.prev_in + (1.0 - a) * frame.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
        self.prev_in = img
        for name in C.STAGE_ORDER:
            if name == "dcp":
                if P["use_dcp"]:
                    t, A = C.estimate_transmission(img, P["omega"], P["t0"], P["patch"])
                    if self.t is not None and self.t.shape == t.shape:
                        wt = np.where(still, w, 0.0).astype(np.float32)          # smooth only where nothing moved
                        t = wt * self.t + (1.0 - wt) * t
                    self.t, self.A = t, self._ema(self.A, A, w)
                    img = C.recover_scene(img, self.t, self.A)
            elif name == "hist":
                if P["hist"] > 0:
                    _, rng = C.histogram_stretch(img, P["hist"], return_range=True)
                    self.hist_range = tuple(self._ema(None if self.hist_range is None else np.array(self.hist_range),
                                                      np.array(rng), w))
                    img = C.histogram_stretch(img, P["hist"], fixed_range=self.hist_range)
            elif name == "tone":
                if C.stage_active("tone", P):
                    if P.get("balance", 0):
                        self.gains = self._ema(self.gains, C.colour_balance_gains(img, P["balance"]), w)
                    img = C.tone_adjust(img, P["gamma"], P["sat"], P.get("balance", 0), gains=self.gains)
            elif C.stage_active(name, P):
                img = C.apply_stage(name, img, P)
        self.P = P
        return img


def flicker(frames_in, frames_out, motion_thr=10.0):
    """Temporal flicker of a processed sequence: mean absolute change between consecutive output frames, in pixels
    that did not change in the input (static background), in grey levels. Lower is steadier."""
    vals = []
    for i in range(1, len(frames_out)):
        gi0, gi1 = (cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in (frames_in[i - 1], frames_in[i]))
        still = cv2.GaussianBlur(np.abs(gi1 - gi0), (0, 0), 1.5) < motion_thr
        go0, go1 = (cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in (frames_out[i - 1], frames_out[i]))
        vals.append(float(np.abs(go1 - go0)[still].mean()))
    return float(np.mean(vals))


def to_h264(path):
    """Re-encode an mp4 as H.264 so that browsers can play it (OpenCV writes MPEG-4 part 2). Needs the optional
    imageio-ffmpeg package; returns True if the file was converted, False (file left as it is) otherwise."""
    try:
        import subprocess
        import imageio_ffmpeg
        tmp = str(path) + ".h264.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", "-preset", "veryfast", tmp], check=True, timeout=600)
        Path(tmp).replace(path)
        return True
    except Exception:
        return False


def process_video(src, dst, P=None, auto=False, temporal=True, max_dim=0, max_frames=0, progress=None, auto_blur=False):
    """Dehaze a video file (any format OpenCV reads) to an mp4. auto: strength follows the haze level; auto_blur: the
    Wiener deblur is set from the blur measured on the first frame. Returns (frames written, seconds per frame)."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dehazer = TemporalDehazer(P, auto=auto, smoothing=0.9 if temporal else 0.0)
    writer, n, t0 = None, 0, time.perf_counter()
    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        s = min(1.0, max_dim / max(bgr.shape[:2])) if max_dim else 1.0
        if s < 1.0:
            bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if auto_blur and n == 0:
            dehazer.base = C.auto_deblur_params(rgb, dehazer.base)[0]
        out = dehazer.process(rgb)
        if writer is None:
            writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
        writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        n += 1
        if progress:
            progress(n, min(n_total, max_frames) if max_frames and n_total else (max_frames or n_total))
    cap.release()
    if writer is not None:
        writer.release()
        to_h264(dst)
    return n, (time.perf_counter() - t0) / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--auto", action="store_true", help="scale the strength to the haze level")
    ap.add_argument("--auto-blur", action="store_true", help="undo the blur measured on the first frame")
    ap.add_argument("--no-temporal", action="store_true", help="process every frame independently (flickers)")
    ap.add_argument("--max-dim", type=int, default=0, help="downscale frames to this long edge")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    n, spf = process_video(a.src, a.dst, auto=a.auto, temporal=not a.no_temporal, max_dim=a.max_dim,
                     max_frames=a.max_frames, auto_blur=a.auto_blur)
    print(f"{n} frames, {spf:.2f} s/frame -> {a.dst}")


if __name__ == "__main__":
    main()
