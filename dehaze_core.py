"""Image-processing core of the Dehaze App: no Streamlit, so it can be tested and
evaluated on its own (see evaluate.py).

Stages (run order): dark channel prior -> Wiener deblur -> adaptive smoothing -> adaptive
sharpening -> histogram stretching -> frequency-domain enhancement -> final tone.
The three course modules are histogram stretching, frequency-domain enhancement, and
adaptive smoothing + sharpening. Every stage returns its input untouched at strength 0.
"""
import hashlib
import math
import random
from collections import OrderedDict

import cv2
import numpy as np
from scipy import ndimage
from scipy.fft import fft2, fftshift, ifft2, ifftshift

FREQ_CUTOFF = 30  # frequency-domain cutoff radius, in frequency-plane pixels

# Default settings. P_BALANCED comes from a parameter search on the outdoor test scenes that have a
# clear reference (all three course modules kept active, tone neutral apart from a mild colour
# balance); see evaluate.py and the README. P_MODULES is the same search with only the three
# course modules allowed.
P_BALANCED = dict(hist=20, freq=20, smooth=23, sharpen=37, omega=0.87, t0=0.20, patch=0, use_dcp=True,
                  deblur=0.0, deblur_kind="Gaussian", deblur_k=0.01, gamma=1.0, sat=1.0, balance=50, cutoff=60)
P_MODULES = dict(P_BALANCED, use_dcp=False, hist=20, freq=20, smooth=20, sharpen=80, balance=0)


def hist_clip_percent(strength):
    return (strength / 100.0) * 2.0          # 0 .. 2 % clipped from each end


def freq_gain(strength):
    return 1.0 + (strength / 100.0) * 1.5    # gain applied to fine detail


def smoothing_tolerance(strength):
    return 10.0 + 0.6 * strength             # how big a colour step still counts as an edge


def edge_gain(strength):
    return (strength / 100.0) * 2.0          # unsharp-mask amount


def _to_ycc(img):
    ycc = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2YCrCb)
    return ycc, ycc[:, :, 0] * 255.0


def _from_ycc(ycc, y):
    out = ycc.copy()
    out[:, :, 0] = np.clip(y, 0.0, 255.0) / 255.0
    rgb = cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def histogram_stretch(img, strength=55, return_range=False):
    """Linear contrast stretch of the luminance, clipping `percent`% at each end."""
    percent = hist_clip_percent(strength)
    ycc, y = _to_ycc(img)
    lo, hi = np.percentile(y, [percent, 100 - percent])
    if strength <= 0 or hi - lo < 1.0:
        out = img
    else:
        out = _from_ycc(ycc, (y - lo) * 255.0 / (hi - lo))
    return (out, (float(lo), float(hi))) if return_range else out


def adaptive_smoothing(img, strength=50):
    """Edge-preserving smoothing: flat areas (and the noise the fog removal
    amplifies there) are smoothed, edges are kept. The strength sets how big a
    colour difference still counts as an edge."""
    if strength <= 0:
        return img
    sigma_color = smoothing_tolerance(strength)
    sigma_space = 3.0 + 0.05 * strength
    return cv2.bilateralFilter(img, 7, sigma_color, sigma_space)


def adaptive_sharpening(img, strength=50):
    """Unsharp mask on luminance, with the added detail capped so strong edges
    don't overshoot into white halos."""
    amount = edge_gain(strength)
    if amount <= 0:
        return img
    ycc, y = _to_ycc(img)
    detail = np.clip(y - cv2.GaussianBlur(y, (0, 0), 1.5), -40.0, 40.0)
    return _from_ycc(ycc, y + amount * detail)


def frequency_enhancement(img, strength=55, cutoff=FREQ_CUTOFF):
    """High-frequency emphasis on luminance: a smooth (Gaussian) high-pass, so
    there is no ringing from a hard cutoff, and the image is mirror-padded so
    the FFT's wrap-around doesn't leak detail across opposite borders."""
    gain = freq_gain(strength)
    if gain <= 1.0:
        return img
    ycc, y = _to_ycc(img)
    h, w = y.shape
    pad = max(0, min(48, h - 1, w - 1))
    yp = np.pad(y, pad, mode="reflect") if pad else y
    rows, cols = yp.shape
    spectrum = fftshift(fft2(yp))
    v = np.arange(rows) - rows // 2
    u = np.arange(cols) - cols // 2
    d2 = v[:, None] ** 2 + u[None, :] ** 2
    high_pass = 1.0 - np.exp(-d2 / (2.0 * cutoff ** 2))
    response = 1.0 + (gain - 1.0) * high_pass      # 1.0 at DC, `gain` at fine detail
    out = np.real(ifft2(ifftshift(spectrum * response)))
    out = out[pad:pad + h, pad:pad + w]
    return _from_ycc(ycc, out)


def _guided_filter(guide, src, radius, eps):
    """Edge-aware smoothing of `src` that follows the edges of `guide`
    (He et al., 'Guided Image Filtering')."""
    k = (2 * radius + 1, 2 * radius + 1)

    def mean(x):
        return cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)

    mean_g, mean_s = mean(guide), mean(src)
    cov = mean(guide * src) - mean_g * mean_s
    var = mean(guide * guide) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return mean(a) * guide + mean(b)


def dark_channel_prior(img, omega=0.95, t0=0.15, patch=0):
    """Single-image haze removal (He et al.): estimate the airlight A and a
    per-pixel transmission t, then recover J = (I - A) / t + A."""
    I = img.astype(np.float32) / 255.0
    h, w = I.shape[:2]
    size = (max(3, int(patch)) | 1) if patch else (max(3, round(max(h, w) / 80)) | 1)  # odd window; auto ~15 px at 1200 px

    dark = ndimage.minimum_filter(I.min(axis=2), size=size)

    # Airlight: of the haziest 0.1% of pixels (brightest in the dark channel),
    # take the single brightest one — one pixel, so its colour stays consistent.
    n = max(1, int(0.001 * dark.size))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    candidates = I.reshape(-1, 3)[idx]
    A = np.maximum(candidates[np.argmax(candidates.sum(axis=1))], 0.1)

    # Transmission, each colour channel normalised by ITS OWN airlight value.
    t = 1.0 - omega * ndimage.minimum_filter((I / A).min(axis=2), size=size)

    # Refine with a guided filter so t follows the photo's real edges. The raw
    # min-filter map is blocky, which shows up as blocks and halos in the result.
    gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    radius = max(1, min(size * 3, (min(h, w) - 1) // 2))
    t = _guided_filter(gray, t.astype(np.float32), radius, 1e-3)
    t = np.clip(t, t0, 1.0)

    J = (I - A) / t[:, :, None] + A
    return np.clip(J * 255.0 + 0.5, 0, 255).astype(np.uint8)


_DCP_CACHE = OrderedDict()


def cached_dark_channel_prior(img, omega=0.95, t0=0.15, patch=0):
    """dark_channel_prior with a small LRU cache (it is the slow stage and rarely changes
    while other sliders move)."""
    key = (hashlib.blake2b(np.ascontiguousarray(img).tobytes(), digest_size=16).hexdigest(),
           img.shape, round(float(omega), 4), round(float(t0), 4), int(patch))
    if key in _DCP_CACHE:
        _DCP_CACHE.move_to_end(key)
        return _DCP_CACHE[key]
    out = dark_channel_prior(img, omega, t0, patch)
    _DCP_CACHE[key] = out
    while len(_DCP_CACHE) > 4:
        _DCP_CACHE.popitem(last=False)
    return out


def tone_adjust(img, gamma=1.0, saturation=1.0, balance=0):
    """Final colour step: gray-world colour balance (0-100 %), saturation and gamma.
    Fog is bluish-grey, so a dehazed image often keeps a colour cast; the balance step pulls the
    channel means back together. All neutral values (balance 0, saturation 1, gamma 1) leave the
    image untouched."""
    if gamma == 1.0 and saturation == 1.0 and not balance:
        return img
    f = img.astype(np.float32) / 255.0
    if balance:
        means = f.reshape(-1, 3).mean(axis=0) + 1e-6
        gains = (means.mean() / means) ** (balance / 100.0)
        f = np.clip(f * gains, 0.0, 1.0)
    if saturation != 1.0:
        lum = (f @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        f = np.clip(lum + (f - lum) * saturation, 0.0, 1.0)
    if gamma != 1.0:
        f = f ** (1.0 / gamma)
    return np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)


STAGE_ORDER = ["dcp", "deblur", "smooth", "sharpen", "hist", "freq", "tone"]   # the order they run


def wiener_deblur(img, amount, kind="Gaussian", k=0.01):
    """Wiener deconvolution of luminance in the Fourier domain, for an assumed Gaussian blur
    (std = amount px) or horizontal motion blur (length = 3 x amount px). k damps noise."""
    if amount <= 0:
        return img
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0].astype(np.float32) / 255.0
    pad = int(4 * amount) + 8
    yp = cv2.copyMakeBorder(y, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    fy = np.fft.fftfreq(yp.shape[0])[:, None]
    fx = np.fft.fftfreq(yp.shape[1])[None, :]
    if kind == "Motion":
        n = max(3, int(round(3 * amount)) | 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            h1 = np.where(fx == 0, 1.0, np.sin(np.pi * n * fx) / (n * np.sin(np.pi * fx)))
        H = np.broadcast_to(h1, yp.shape)
    else:
        H = np.exp(-2.0 * np.pi ** 2 * amount ** 2 * (fx ** 2 + fy ** 2))
    X = np.fft.fft2(yp) * H / (H * H + k)
    x = np.real(np.fft.ifft2(X))[pad:-pad, pad:-pad]
    ycc[..., 0] = np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def apply_stage(name, img, P):
    if name == "dcp":
        return cached_dark_channel_prior(img, P["omega"], P["t0"], P["patch"]) if P["use_dcp"] else img
    if name == "deblur":
        return wiener_deblur(img, P["deblur"], P["deblur_kind"], P["deblur_k"])
    if name == "smooth":
        return adaptive_smoothing(img, P["smooth"])
    if name == "sharpen":
        return adaptive_sharpening(img, P["sharpen"])
    if name == "hist":
        return histogram_stretch(img, P["hist"])
    if name == "freq":
        return frequency_enhancement(img, P["freq"], P["cutoff"])
    return tone_adjust(img, P["gamma"], P["sat"], P.get("balance", 0))


def stage_active(name, P):
    return {"dcp": P["use_dcp"], "deblur": P["deblur"] > 0, "smooth": P["smooth"] > 0,
            "sharpen": P["sharpen"] > 0, "hist": P["hist"] > 0, "freq": P["freq"] > 0,
            "tone": P["gamma"] != 1.0 or P["sat"] != 1.0 or bool(P.get("balance", 0))}[name]


def dehaze_pipeline(img, P):
    for name in STAGE_ORDER:
        img = apply_stage(name, img, P)
    return img


FOG_AIRLIGHT = np.array([216, 223, 229], dtype=np.float32)


def make_sample_scene(w=640, h=400):
    """Synthetic city street plus its depth map (0 = near, 1 = far), so fog can be
    simulated with the real scattering model and the clear scene is known."""
    k = w / 512.0
    rnd = random.Random(11)
    hz = int(h * 0.56)
    img = np.zeros((h, w, 3), np.uint8)
    depth = np.ones((h, w), np.float32)
    t = np.linspace(0, 1, hz)[:, None, None]
    img[:hz] = (np.array([58, 108, 176]) * (1 - t) + np.array([211, 227, 242]) * t).astype(np.uint8)
    img[hz:] = (60, 63, 67)
    depth[hz:] = np.linspace(0.74, 0.04, h - hz)[:, None]

    def buildings(dv, col, wmin, wr, hmin, hr, x0, x1):
        col = np.array(col, np.float32)
        win = tuple(int(c) for c in (col * 0.5 + np.array([255, 235, 170]) * 0.5))
        bx = x0
        while bx < x1:
            bw, bh = (wmin + rnd.random() * wr) * k, (hmin + rnd.random() * hr) * k
            xa, xb, ya, yb = int(bx), int(bx + bw), int(hz - bh), hz + int(8 * k)
            cv2.rectangle(img, (xa, ya), (xb, yb), tuple(int(c) for c in col), -1)
            cv2.rectangle(depth, (xa, ya), (xb, yb), float(dv), -1)
            for wy in range(ya + int(8 * k), hz - int(6 * k), int(12 * k)):
                for wx in range(xa + int(5 * k), xb - int(6 * k), int(9 * k)):
                    if rnd.random() > 0.45:
                        cv2.rectangle(img, (wx, wy), (wx + int(4 * k), wy + int(6 * k)), win, -1)
            bx += bw + (2 + rnd.random() * 4) * k

    buildings(0.90, (109, 129, 150), 26, 50, 60, 110, int(-10 * k), w)
    buildings(0.65, (78, 95, 114), 34, 50, 50, 150, int(-10 * k), w)
    buildings(0.42, (53, 66, 79), 40, 40, 40, 170, int(-10 * k), w)
    cv2.fillPoly(img, [np.array([[w * .46, hz], [w * .54, hz], [w * 1.1, h], [-w * .1, h]], np.int32)], (37, 39, 43))
    for i in range(1, 9):
        tt = i / 9
        py, sw = hz + (h - hz) * tt * tt, (1 + tt * 10) * k
        cv2.rectangle(img, (int(w / 2 - sw / 2), int(py)), (int(w / 2 + sw / 2), int(py + (2 + tt * 14) * k)),
                      (232, 211, 74), -1)
    buildings(0.12, (29, 36, 44), 70, 60, 170, 90, int(-20 * k), int(120 * k))
    buildings(0.12, (29, 36, 44), 70, 60, 170, 90, int(w - 130 * k), int(w + 20 * k))
    for cx, cy, sc, col in [(200, hz / k + 40, .6, (179, 38, 30)), (300, hz / k + 62, .9, (216, 216, 216)),
                            (110, hz / k + 104, 1.3, (43, 95, 168)), (330, hz / k + 120, 1.5, (224, 162, 26))]:
        x0, y0, s_ = int(cx * k), int(cy * k), sc * k
        cv2.rectangle(img, (x0, y0), (int(x0 + 60 * s_), int(y0 + 16 * s_)), col, -1)
        cv2.rectangle(img, (int(x0 + 12 * s_), int(y0 - 10 * s_)), (int(x0 + 48 * s_), int(y0 + s_)), col, -1)
        for wx in (6, 44):
            cv2.rectangle(img, (int(x0 + wx * s_), int(y0 + 13 * s_)), (int(x0 + (wx + 10) * s_), int(y0 + 20 * s_)), (17, 17, 17), -1)
        cv2.rectangle(depth, (x0, int(y0 - 10 * s_)), (int(x0 + 60 * s_), int(y0 + 20 * s_)), 0.157, -1)
    return img, depth


def simulate_fog(img, depth, density):
    """Atmospheric scattering model I = J·t + A·(1 − t), t = exp(−β·d). Photos have no depth
    map, so depth is guessed to grow toward the top of the frame."""
    h, w = img.shape[:2]
    if depth is None:
        depth = np.repeat(np.linspace(0.95, 0.15, h, dtype=np.float32)[:, None], w, axis=1)
    t = np.exp(-float(density) * depth)[..., None]
    return np.clip(img.astype(np.float32) * t + FOG_AIRLIGHT * (1 - t) + 0.5, 0, 255).astype(np.uint8)


def simulate_blur(img, amount, kind):
    """Camera blur: Gaussian, or horizontal motion (length = 3 x amount px)."""
    if amount <= 0:
        return img
    if kind == "Motion":
        n = max(3, int(round(3 * amount)) | 1)
        kern = np.zeros((n, n), np.float32)
        kern[n // 2, :] = 1.0 / n
        return cv2.filter2D(img, -1, kern, borderType=cv2.BORDER_REFLECT_101)
    return cv2.GaussianBlur(img, (0, 0), float(amount))


def edge_maps(a, b):
    """Sobel gradient magnitude of both images on one shared scale."""
    def mag(img):
        L = _lum(img)
        return np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    ma, mb = mag(a), mag(b)
    top = max(float(np.percentile(np.concatenate([ma.ravel(), mb.ravel()]), 99.5)), 1e-6)
    return [np.clip(m / top * 255.0, 0, 255).astype(np.uint8) for m in (ma, mb)]


def _lum(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def quality_metrics(img):
    """No-reference measures of how much structure an image shows."""
    L = _lum(img)
    hist = np.bincount(L.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    mag = np.hypot(L[:-1, 1:] - L[:-1, :-1], L[1:, :-1] - L[:-1, :-1])
    chans = [np.bincount(img[..., c].ravel(), minlength=256).astype(np.float64) for c in range(3)]
    return {"contrast": float(L.std()), "entropy": float(-(p * np.log2(p)).sum()),
            "gradient": float(mag.mean()), "edges": float(100.0 * (mag > 12).mean()),
            "hist": hist, "chans": chans}


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-6 else 10.0 * math.log10(255.0 ** 2 / mse)


def ssim(a, b):
    """Mean structural similarity on luminance (11x11 Gaussian window)."""
    x, y = _lum(a), _lum(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def g(z):
        return cv2.GaussianBlur(z, (11, 11), 1.5)

    mx, my = g(x), g(y)
    sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean())
