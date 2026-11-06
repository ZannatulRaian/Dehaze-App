"""Quick checks for dehaze_core.py.   python test_core.py"""
import cv2
import numpy as np

import dehaze_core as C
from evaluate import load_pairs, load_urban

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
# 6. blur estimation: sharp photos read as sharp (also in fog); known blur is found, with its type, within 30 %
assert C.estimate_blur(clear)["kind"] == "none" and C.estimate_blur(foggy)["kind"] == "none"
for kind, amount in (("Gaussian", 3.0), ("Motion", 3.0)):
    est = C.estimate_blur(C.simulate_blur(foggy, amount, kind))
    assert est["kind"] == kind and abs(est["amount"] - amount) < 0.3 * amount, (kind, est)
assert C.estimate_blur(np.full((120, 160, 3), 128, np.uint8))["kind"] == "none"      # no edges at all
assert C.estimate_blur(img[:20, :20])["kind"] == "none"                               # too small to measure
P_auto, est = C.auto_deblur_params(C.simulate_blur(clear, 3.0, "Motion"), C.P_BALANCED)
assert P_auto["deblur"] > 0 and P_auto["deblur_kind"] == "Motion"
assert C.auto_deblur_params(clear, C.P_BALANCED)[0]["deblur"] == 0.0

# 7. urban test set: on disk it matches the generator, depth is a 0..1 map, and the dark channel's
#    transmission follows the true one
urban = load_urban()
assert len(urban) >= 6
c0, d0 = next(iter(urban.values()))
assert c0.shape[:2] == d0.shape and 0.0 <= d0.min() and d0.max() <= 1.0
assert np.array_equal(c0, C.make_urban_set(1)[0][0])
t_est, A = C.estimate_transmission(C.simulate_fog(c0, d0, 1.6), 0.87, 0.2, 0)
assert np.corrcoef(t_est.ravel(), np.clip(C.true_transmission(d0, 1.6), 0.2, 1).ravel())[0, 1] > 0.8
assert np.array_equal(C.dark_channel_prior(foggy, 0.87, 0.2), C.dark_channel_prior(foggy, 0.87, 0.2))

# 8. downstream test: fog hides edges and lines, dehazing brings them back; a map matches itself perfectly
fog_c = C.simulate_fog(c0, d0, 3.0)
dn_fog, dn_clear = C.downstream_metrics(fog_c, c0), C.downstream_metrics(c0, c0)
dn_out = C.downstream_metrics(C.dehaze_pipeline(fog_c, C.P_BALANCED), c0)
assert dn_out["edge_f1"] > dn_fog["edge_f1"] and dn_out["line_recall"] > dn_fog["line_recall"]
assert dn_clear["lines_vertical"] > 0 and dn_clear["lines_oblique"] > 0                # building and lane lines exist
assert abs(dn_clear["edge_f1"] - 1.0) < 1e-9 and dn_clear["line_recall"] == 1.0
assert C.edge_f1(np.zeros((9, 9), np.uint8), np.zeros((9, 9), np.uint8))[2] == 0.0     # empty maps do not crash
assert len(C.hough_lines(np.zeros((50, 50), np.uint8))) == 0
print("all checks passed")
