"""Quick checks for dehaze_core.py.   python test_core.py"""
import cv2
import numpy as np

import dehaze_core as C
from evaluate import load_pairs

rng = np.random.RandomState(0)
img = rng.randint(0, 256, (97, 131, 3), dtype=np.uint8)          # odd sizes on purpose
OFF = dict(C.P_BALANCED, use_dcp=False, hist=0, freq=0, smooth=0, sharpen=0, balance=0)

# 1. every stage leaves the image untouched when switched off
for name in C.STAGE_ORDER:
    assert not C.stage_active(name, OFF), name
    assert np.array_equal(C.apply_stage(name, img, OFF), img), f"{name} changed the image while off"

# 2. the full pipeline keeps shape and dtype
out = C.dehaze_pipeline(img, C.P_BALANCED)
assert out.shape == img.shape and out.dtype == np.uint8

# 3. simulated fog flattens edges (lower mean gradient); Wiener deblur undoes simulated blur
pairs = load_pairs(480)
clear = next(iter(pairs.values()))[1]
foggy = C.simulate_fog(clear, None, 1.6)
assert C.quality_metrics(foggy)["gradient"] < C.quality_metrics(clear)["gradient"]
for kind in ("Gaussian", "Motion"):
    blurred = C.simulate_blur(clear, 2.0, kind)
    restored = C.wiener_deblur(blurred, 2.0, kind, 0.003)
    assert C.psnr(restored, clear) > C.psnr(blurred, clear) + 1.0, kind

# 4. on the real foggy test photos the default settings beat doing nothing (PSNR and SSIM)
none_s = [C.ssim(h, c) for h, c in pairs.values()]
none_p = [C.psnr(h, c) for h, c in pairs.values()]
full_s = [C.ssim(C.dehaze_pipeline(h, C.P_BALANCED), c) for h, c in pairs.values()]
full_p = [C.psnr(C.dehaze_pipeline(h, C.P_BALANCED), c) for h, c in pairs.values()]
assert np.mean(full_s) > np.mean(none_s) and np.mean(full_p) > np.mean(none_p)

# 5. the metrics themselves
assert abs(C.ssim(img, img) - 1.0) < 1e-6 and C.psnr(img, img) == 99.0
assert cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).shape == img.shape[:2]
print("all checks passed")
