"""Vehicle and pedestrian detection for the downstream test (evaluation only, not used by the app).

Runs YOLOv5n (COCO, ONNX) through onnxruntime. The weights are NOT bundled: they are downloaded once to
./models/ on first use (https://github.com/ultralytics/yolov5/releases/tag/v7.0, AGPL-3.0 licence).

    import detect
    boxes = detect.detect(img_rgb)                  # [(x1, y1, x2, y2, score, class_name), ...]
    detect.match(boxes, ref_boxes)                  # recall / precision against the clear photo's detections
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
MODEL_PATH = Path(__file__).parent / "models" / "yolov5n.onnx"
# COCO classes that matter for traffic monitoring: id -> name
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLES = {"bicycle", "car", "motorcycle", "bus", "truck"}
_session = None


def available():
    """True if the model is present or can be downloaded."""
    try:
        _load()
        return True
    except Exception:
        return False


def _load():
    global _session
    if _session is None:
        import onnxruntime as ort
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def detect(img, conf=0.25, iou=0.45, classes=None):
    """Detections of people and vehicles in an RGB image: list of (x1, y1, x2, y2, score, class_name)."""
    sess = _load()
    h, w = img.shape[:2]
    s = 640.0 / max(h, w)
    nh, nw = round(h * s), round(w * s)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    top, left = (640 - nh) // 2, (640 - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    if "float16" in sess.get_inputs()[0].type:                            # the released yolov5n.onnx is half precision
        x = x.astype(np.float16)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0].astype(np.float32)   # (25200, 85)
    keep = classes or CLASSES
    scores = out[:, 4:5] * out[:, 5:]
    cls = scores.argmax(1)
    best = scores[np.arange(len(out)), cls]
    sel = (best >= conf) & np.isin(cls, list(keep))
    if not sel.any():
        return []
    b, sc, cl = out[sel, :4], best[sel], cls[sel]
    xyxy = np.stack([(b[:, 0] - b[:, 2] / 2 - left) / s, (b[:, 1] - b[:, 3] / 2 - top) / s,
                     (b[:, 0] + b[:, 2] / 2 - left) / s, (b[:, 1] + b[:, 3] / 2 - top) / s], 1)
    wh = [[float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])] for r in xyxy]
    idx = cv2.dnn.NMSBoxes(wh, sc.tolist(), conf, iou)
    return [(*np.clip(xyxy[i], 0, [w, h, w, h]).tolist(), float(sc[i]), keep[int(cl[i])]) for i in np.array(idx).ravel()]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(boxes, ref_boxes, thr=0.5):
    """(recall, precision, n_matched): a reference detection counts as found when a detection of the same
    class overlaps it with IoU >= thr (each detection is used once)."""
    used, hit = set(), 0
    for r in ref_boxes:
        best, bi = 0.0, None
        for i, b in enumerate(boxes):
            if i not in used and b[5] == r[5] and _iou(b, r) > max(best, thr - 1e-9):
                best, bi = _iou(b, r), i
        if bi is not None:
            used.add(bi)
            hit += 1
    return (hit / len(ref_boxes) if ref_boxes else float("nan"),
            hit / len(boxes) if boxes else float("nan"), hit)
