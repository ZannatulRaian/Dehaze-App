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
# For traffic and site monitoring, where a detector or an operator looks for vehicles and people: fog removal plus
# histogram stretching only. Frequency enhancement, sharpening and gray-world colour balance each lowered vehicle detection
# recall on real street photos (see evaluate.py, ablation and section 2) although they raise edge counts.
P_TRAFFIC = dict(P_BALANCED, hist=20, freq=0, smooth=0, sharpen=0, balance=0)


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


def histogram_stretch(img, strength=55, return_range=False, fixed_range=None):
    """Linear contrast stretch of the luminance, clipping `percent`% at each end. `fixed_range` = (lo, hi) uses
    that range instead of the image's own (video uses a range smoothed over time)."""
    percent = hist_clip_percent(strength)
    ycc, y = _to_ycc(img)
    lo, hi = fixed_range if fixed_range is not None else np.percentile(y, [percent, 100 - percent])
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


def estimate_transmission(img, omega=0.95, t0=0.15, patch=0):
    """Dark-channel estimate of the airlight A and the per-pixel transmission t (guided-filter
    refined, clipped to [t0, 1]). Returned separately so it can be checked against a known depth map."""
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
    return np.clip(t, t0, 1.0), A


def dark_channel_prior(img, omega=0.95, t0=0.15, patch=0):
    """Single-image haze removal (He et al.): estimate the airlight A and a
    per-pixel transmission t, then recover J = (I - A) / t + A."""
    t, A = estimate_transmission(img, omega, t0, patch)
    return recover_scene(img, t, A)


def recover_scene(img, t, A):
    """J = (I - A) / t + A for a given transmission map t and airlight A."""
    J = (img.astype(np.float32) / 255.0 - A) / t[:, :, None] + A
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


def colour_balance_gains(img, balance):
    """Per-channel gains of the gray-world colour balance (balance = 0-100 %)."""
    means = (img.astype(np.float32) / 255.0).reshape(-1, 3).mean(axis=0) + 1e-6
    return (means.mean() / means) ** (balance / 100.0)


def tone_adjust(img, gamma=1.0, saturation=1.0, balance=0, gains=None):
    """Final colour step: gray-world colour balance (0-100 %), saturation and gamma.
    Fog is bluish-grey, so a dehazed image often keeps a colour cast; the balance step pulls the
    channel means back together. All neutral values (balance 0, saturation 1, gamma 1) leave the
    image untouched."""
    if gamma == 1.0 and saturation == 1.0 and not balance:
        return img
    f = img.astype(np.float32) / 255.0
    if balance:
        if gains is None:                      # `gains` lets a video reuse colour-balance gains smoothed over time
            gains = colour_balance_gains(img, balance)
        f = np.clip(f * gains, 0.0, 1.0)
    if saturation != 1.0:
        lum = (f @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        f = np.clip(lum + (f - lum) * saturation, 0.0, 1.0)
    if gamma != 1.0:
        f = f ** (1.0 / gamma)
    return np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)


STAGE_ORDER = ["dcp", "deblur", "smooth", "sharpen", "hist", "freq", "tone"]   # the order they run


def motion_kernel(n, angle=0.0):
    """Normalised line kernel of length n px at `angle` degrees (0 = horizontal, 90 = vertical, counter-clockwise
    on screen), anti-aliased by drawing it 8x larger. Angle 0 is exactly the n-tap box used for horizontal blur."""
    n = max(3, int(n) | 1)
    if angle % 180.0 == 0.0:
        kern = np.zeros((n, n), np.float32)
        kern[n // 2, :] = 1.0 / n
        return kern
    big, c = n * 8, (n * 8) // 2
    dx, dy = math.cos(math.radians(angle)) * (n * 4 - 4), -math.sin(math.radians(angle)) * (n * 4 - 4)
    canvas = np.zeros((big, big), np.float32)
    cv2.line(canvas, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1.0, 8, cv2.LINE_AA)
    kern = cv2.resize(canvas, (n, n), interpolation=cv2.INTER_AREA)
    return kern / kern.sum()


def wiener_deblur(img, amount, kind="Gaussian", k=0.01, angle=0.0):
    """Wiener deconvolution of luminance in the Fourier domain, for an assumed Gaussian blur
    (std = amount px) or linear motion blur (length = 3 x amount px, at `angle` degrees, 0 = horizontal).
    k damps noise."""
    if amount <= 0:
        return img
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0].astype(np.float32) / 255.0
    pad = int(4 * amount * (1.5 if kind == "Motion" and angle % 180.0 else 1.0)) + 8
    yp = cv2.copyMakeBorder(y, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    fy = np.fft.fftfreq(yp.shape[0])[:, None]
    fx = np.fft.fftfreq(yp.shape[1])[None, :]
    if kind == "Motion" and angle % 180.0 != 0.0:
        kern = motion_kernel(int(round(3 * amount)), angle)
        psf = np.zeros(yp.shape, np.float32)
        kh, kw = kern.shape
        psf[:kh, :kw] = kern
        H = np.real(np.fft.fft2(np.roll(psf, (-(kh // 2), -(kw // 2)), axis=(0, 1))))   # centred kernel -> real response
    elif kind == "Motion":
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



# ---------------------------------------------------------------- haze level and automatic strength
HAZE_SLOPE, HAZE_OFFSET, STRENGTH_MIN = 2.1, 0.35, 0.25   # strength = clip(2.1 * haze - 0.35, 0.25, 1); fit in evaluate.py


def haze_level(img):
    """How hazy a photo is, 0 (clear) .. 1: 1 - mean dark-channel transmission. It rises with fog density in every
    kind of scene tested, but its level differs between scenes, so it is used to scale the processing strength
    (strength_scale) rather than read as a fog density."""
    t, _ = estimate_transmission(img, 0.95, 0.05, 0)
    return float(1.0 - t.mean())


def strength_scale(haze):
    """Share of the base strength to apply: 0.25 on a near-clear photo up to 1.0 (the full default) in dense fog."""
    return float(np.clip(HAZE_SLOPE * haze - HAZE_OFFSET, STRENGTH_MIN, 1.0))


def auto_strength_params(img, P=None, haze=None):
    """Copy of the settings P (default: Balanced) with the strengths of histogram stretching, frequency enhancement,
    smoothing, sharpening and the dark channel's omega scaled to the photo's haze level. Returns (settings, haze, scale).
    Fitted so that light haze is not over-processed: fixed settings tuned for dense fog lower SSIM on light fog."""
    P = dict(P_BALANCED if P is None else P)
    h = haze_level(img) if haze is None else haze        # `haze`: a level already measured (the app caches it)
    g = strength_scale(h)
    for k in ("hist", "freq", "smooth", "sharpen"):
        P[k] = P[k] * g
    if P["use_dcp"]:
        P["omega"] = min(0.95, P["omega"] * (0.35 + 0.65 * g))
    return P, h, g


# ---------------------------------------------------------------- blur estimation
BLUR_DIRECTIONS = 8    # profile directions measured, every 22.5 degrees
BLUR_BASELINE_AXIS, BLUR_BASELINE_DIAG = 0.65, 0.72   # edge spread (px, std) of a sharp photo along / between the image axes
BLUR_MIN = 0.7         # excess spread (px) below which an image is called sharp
BLUR_ANISOTROPY = 0.32  # 1 - mean/peak of the excess variance: above it the blur is directional (motion), below it round
GAUSS_SCALE = 1.08     # calibration factors: the raw estimates read about 8 % (Gaussian) and 10 % (motion) low
MOTION_SCALE = 1.10    # (the tails of the edge profile are cut); fit on the 7 real photos only, so the synthetic
                       # cities stay an independent check (see evaluate.py, blur estimation)


def _edge_spread(L, axis, top=0.08, reach=30):
    """Median spread (std, px) of the intensity profile across the strongest edges, measured along one
    axis (1 = across columns, i.e. horizontal profiles). A step edge blurred by a kernel has a gradient
    profile equal to that kernel, so its std is the kernel's std: sigma for Gaussian blur, n/sqrt(12)
    for a box of length n (horizontal motion blur). Fog scales the step and the gradient by the same
    factor, so the spread does not depend on fog density. Returns None if there are too few edges."""
    if axis == 0:
        L = L.T
    h, w = L.shape
    if w < 2 * reach + 3 or h < 8:
        return None
    g = np.zeros_like(L)
    g[:, 1:-1] = (L[:, 2:] - L[:, :-2]) * 0.5
    a = np.abs(g)
    peak = (a >= np.roll(a, 1, 1)) & (a > np.roll(a, -1, 1)) & (a > 0)
    peak[:, :reach + 1] = peak[:, w - reach - 1:] = False
    if peak.sum() < 100:
        return None
    ys, xs = np.nonzero(peak & (a >= np.quantile(a[peak], 1.0 - top)))
    sgn, pk = np.sign(g[ys, xs]), a[ys, xs]
    m0, m1, m2 = pk.copy(), np.zeros(len(xs)), np.zeros(len(xs))
    for d in (1, -1):                            # walk out of the edge while the gradient keeps its sign
        alive = np.ones(len(xs), bool)
        for k in range(1, reach):
            v = g[ys, xs + d * k] * sgn
            alive &= v > 0.05 * pk
            wk = np.where(alive, v, 0.0)
            m0 += wk
            m1 += wk * d * k
            m2 += wk * k * k
    c = m1 / m0
    return float(np.median(np.sqrt(np.maximum(m2 / m0 - c * c, 0.0))))


def _spread_along(L, angle):
    """Edge spread of intensity profiles measured along direction `angle` (degrees, 0 = along x, counter-clockwise
    on screen); other directions than the axes are measured on the image rotated so that direction is horizontal."""
    if angle % 180.0 == 0.0:
        return _edge_spread(L, 1)
    if angle % 180.0 == 90.0:
        return _edge_spread(L, 0)
    h, w = L.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
    return _edge_spread(cv2.warpAffine(L, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), 1)


def estimate_blur(img):
    """Blind estimate of the blur in a photo, in the units `wiener_deblur` and `simulate_blur` use.

    The spread of the strongest edges is measured along 8 directions and the width of a sharp photo is
    subtracted in quadrature. A blur that is as wide in every direction is Gaussian (amount = sigma); one with a
    clear peak in one direction is linear motion blur: the peak's direction (refined between the measured
    directions) is the angle and its excess spread gives the length (amount = length / 3).
    Returns {"kind": "none" | "Gaussian" | "Motion", "amount": px, "angle": degrees (motion only, 0 = horizontal,
    90 = vertical), "spreads": the 8 measured spreads}."""
    L = _lum(img)
    n = BLUR_DIRECTIONS
    angles = [k * 180.0 / n for k in range(n)]
    sp = [_spread_along(L, a) for a in angles]
    out = {"kind": "none", "amount": 0.0, "angle": 0.0, "spreads": sp}
    if any(x is None for x in sp):
        return out
    base = np.array([BLUR_BASELINE_AXIS if a % 90.0 == 0.0 else BLUR_BASELINE_DIAG for a in angles])
    v = np.maximum(np.array(sp) ** 2 - base ** 2, 0.0)             # excess variance per direction
    k = int(v.argmax())
    vm, v0, vp = v[(k - 1) % n], v[k], v[(k + 1) % n]
    den = vm - 2.0 * v0 + vp
    off = float(np.clip(0.5 * (vm - vp) / den, -0.5, 0.5)) if den < 0 else 0.0   # parabola through the peak
    peak = v0 - 0.25 * (vm - vp) * off if den < 0 else v0
    if math.sqrt(max(peak, 0.0)) < BLUR_MIN:
        return out
    if 1.0 - v.mean() / peak >= BLUR_ANISOTROPY:                    # directional: linear motion blur
        out.update(kind="Motion", angle=round(((k + off) * 180.0 / n) % 180.0, 1),
                   amount=MOTION_SCALE * math.sqrt(12.0 * peak) / 3.0)   # box of length L: variance L^2/12, L = 3 * amount
    else:
        out.update(kind="Gaussian", amount=GAUSS_SCALE * math.sqrt(v.mean()))
    out["amount"] = round(out["amount"], 2)
    return out


def auto_deblur_params(img, P):
    """Copy of the settings P with the Wiener deblur set from `estimate_blur(img)`
    (switched off when the photo looks sharp)."""
    est = estimate_blur(img)
    if est["kind"] in ("Gaussian", "Motion"):
        return dict(P, deblur=est["amount"], deblur_kind=est["kind"], deblur_angle=est["angle"]), est
    return dict(P, deblur=0.0), est


def apply_stage(name, img, P):
    if name == "dcp":
        return cached_dark_channel_prior(img, P["omega"], P["t0"], P["patch"]) if P["use_dcp"] else img
    if name == "deblur":
        return wiener_deblur(img, P["deblur"], P["deblur_kind"], P["deblur_k"], P.get("deblur_angle", 0.0))
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


def make_sample_scene(w=640, h=400, seed=11, lane_lines=False):
    """Synthetic city street plus its depth map (0 = near, 1 = far), so fog can be
    simulated with the real scattering model and the clear scene is known. `seed` changes the
    building layout (11 is the scene the app shows); `lane_lines` adds two solid lane lines, which the
    evaluation's line detector needs (the dashes alone are too short to count as lines)."""
    k = w / 512.0
    rnd = random.Random(seed)
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
    if lane_lines:
        for f in (0.2, 0.8):          # solid lane lines converging on the vanishing point
            cv2.line(img, (int(w * (0.46 + 0.08 * f)), hz), (int(-0.1 * w + 1.2 * w * f), h), (225, 225, 225),
                     max(1, int(2 * k)), cv2.LINE_AA)
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


def make_urban_set(n=6, w=640, h=400):
    """n synthetic city streets (different building layouts), each as (clear image, true depth map)."""
    return [make_sample_scene(w, h, seed=11 + i, lane_lines=True) for i in range(n)]


def make_traffic_clip(n=30, w=640, h=400, seed=11, cars=3):
    """A short clip of a fixed street camera: the synthetic city (with lane lines) and `cars` cars driving across the
    road at different speeds. Returns (clear frames, depth map); the depth of the static scene is reused for the cars
    (they drive on the road, so each pixel keeps the road's depth). Frames are RGB uint8."""
    base, depth = make_sample_scene(w, h, seed=seed, lane_lines=True)
    rnd = random.Random(seed + 100)
    specs = []
    for i in range(cars):
        y = int(h * (0.72 + 0.09 * i))                       # lower on screen = nearer = bigger
        cw = int(w * (0.07 + 0.04 * i))
        specs.append(dict(y=y, w=cw, h=int(cw * 0.5), x0=rnd.uniform(0, w), v=(3.0 + 2.0 * i) * (1 if i % 2 == 0 else -1),
                          color=[(180, 40, 40), (40, 70, 160), (200, 200, 205)][i % 3]))
    frames = []
    for f in range(n):
        im = base.copy()
        for c in specs:
            x = int((c["x0"] + c["v"] * f) % (w + c["w"])) - c["w"] // 2
            y0 = c["y"] - c["h"]
            cv2.rectangle(im, (x, y0), (x + c["w"], c["y"]), c["color"], -1)
            cv2.rectangle(im, (x + c["w"] // 6, y0 + c["h"] // 8), (x + c["w"] * 5 // 6, y0 + c["h"] * 5 // 12), (30, 35, 45), -1)
            for wx in (x + c["w"] // 5, x + c["w"] * 4 // 5):
                cv2.circle(im, (wx, c["y"]), max(2, c["h"] // 6), (15, 15, 15), -1)
        frames.append(im)
    return frames, depth


def true_transmission(depth, density):
    """Transmission the scattering model gives for a known depth map: t = exp(-beta * d)."""
    return np.exp(-float(density) * depth).astype(np.float32)


def simulate_fog(img, depth, density):
    """Atmospheric scattering model I = J·t + A·(1 − t), t = exp(−β·d). Photos have no depth
    map, so depth is guessed to grow toward the top of the frame."""
    h, w = img.shape[:2]
    if depth is None:
        depth = np.repeat(np.linspace(0.95, 0.15, h, dtype=np.float32)[:, None], w, axis=1)
    t = np.exp(-float(density) * depth)[..., None]
    return np.clip(img.astype(np.float32) * t + FOG_AIRLIGHT * (1 - t) + 0.5, 0, 255).astype(np.uint8)


def simulate_blur(img, amount, kind, angle=0.0):
    """Camera blur: Gaussian, or linear motion (length = 3 x amount px, at `angle` degrees, 0 = horizontal)."""
    if amount <= 0:
        return img
    if kind == "Motion":
        return cv2.filter2D(img, -1, motion_kernel(int(round(3 * amount)), angle), borderType=cv2.BORDER_REFLECT_101)
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


def visibility_metrics(orig, restored, contrast=0.05):
    """No-reference visibility indicators for fog removal (Hautiere et al.), comparing a restored image with the
    foggy original it came from. Visible edges are pixels whose 3x3 neighbourhood spans at least 5 % of the range.
    e: relative gain in visible edges, (n_restored - n_orig) / n_orig.
    r: geometric mean of the gradient ratio restored/original on the restored image's visible edges (> 1 = contrast gained).
    sigma: share of pixels that are black or white in the restored image but were not in the original (clipping)."""
    lo, hi = _lum(orig), _lum(restored)
    def visible(L):
        k = np.ones((3, 3), np.uint8)
        return (cv2.dilate(L, k) - cv2.erode(L, k)) >= contrast * 255.0
    vo, vr = visible(lo), visible(hi)
    n0, nr = int(vo.sum()), int(vr.sum())
    grad = lambda L: np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    go, gr = grad(lo), grad(hi)
    ratio = (gr[vr] + 1e-3) / (go[vr] + 1e-3)
    clip = lambda L: (L <= 1) | (L >= 254)
    return {"e": (nr - n0) / max(n0, 1), "r": float(np.exp(np.log(ratio).mean())) if nr else float("nan"),
            "sigma": float((clip(hi) & ~clip(lo)).mean())}


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


# ---------------------------------------------------------------- downstream test: edges and lines
CANNY_LOW, CANNY_HIGH = 50, 150    # the textbook fixed thresholds, so fog's low contrast shows up


def canny_edges(img, auto=False):
    """Standard Canny edge map (5x5 Gaussian pre-blur). auto=False: fixed thresholds 50/150;
    auto=True: thresholds set from the image's median (0.66 and 1.33 x median), which hides low contrast."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 1.4)
    if auto:
        m = float(np.median(gray))
        return cv2.Canny(gray, max(5, 0.66 * m), max(15, 1.33 * m))
    return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)


def hough_lines(edges):
    """Straight segments (x1, y1, x2, y2) from the probabilistic Hough transform, shape (n, 4)."""
    cv2.setRNGSeed(0)                                  # HoughLinesP is randomised; fix it so runs repeat
    found = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=8)
    return np.zeros((0, 4), np.float32) if found is None else found.reshape(-1, 4).astype(np.float32)


def line_angles(lines):
    """Angle of each segment from horizontal in degrees, 0 .. 180."""
    return np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])) % 180.0


def edge_f1(edges, ref_edges, tol=2):
    """Precision, recall and F1 of an edge map against a reference edge map, counting a pixel as
    matched when the other map has an edge within `tol` pixels."""
    def near(a, b):      # share of edge pixels of a that have an edge of b within tol
        if not a.any():
            return 0.0
        dist = cv2.distanceTransform((b == 0).astype(np.uint8), cv2.DIST_L2, 3)
        return float((dist[a > 0] <= tol).mean())
    pr, rc = near(edges, ref_edges) if ref_edges.any() else 0.0, near(ref_edges, edges)
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def line_recall(lines, ref_lines, tol=4.0, max_angle=5.0):
    """Share of the reference segments that have a found segment within `max_angle` degrees whose
    line passes within `tol` px of the reference segment's midpoint."""
    if len(ref_lines) == 0:
        return float("nan")
    if len(lines) == 0:
        return 0.0
    mid = (ref_lines[:, :2] + ref_lines[:, 2:]) * 0.5
    ang_d = np.abs(line_angles(ref_lines)[:, None] - line_angles(lines)[None, :])
    ang_d = np.minimum(ang_d, 180.0 - ang_d)
    p, q = lines[:, :2], lines[:, 2:]
    d = q - p
    norm = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
    off = np.abs((mid[:, None, 0] - p[None, :, 0]) * d[None, :, 1]
                 - (mid[:, None, 1] - p[None, :, 1]) * d[None, :, 0]) / norm[None, :]
    along = ((mid[:, None, 0] - p[None, :, 0]) * d[None, :, 0] + (mid[:, None, 1] - p[None, :, 1]) * d[None, :, 1]) / norm[None, :]
    ok = (ang_d <= max_angle) & (off <= tol) & (along >= -tol) & (along <= norm[None, :] + tol)
    return float(ok.any(axis=1).mean())


def downstream_metrics(img, ref=None):
    """Simple downstream test: what a standard edge detector and line detector find in the photo.
    Counts: Canny edge pixels (% of the image), Hough segments, of which near-vertical ones (within 15 deg
    of vertical: building edges) and oblique ones (20-70 deg from horizontal: lane lines, roof lines).
    With a clear reference `ref` also: edge F1 against the reference's edges (fixed and auto thresholds) and
    the share of the reference's line segments that are found again."""
    edges = canny_edges(img)
    lines = hough_lines(edges)
    ang = line_angles(lines)
    out = {"canny_pct": float(100.0 * (edges > 0).mean()), "lines": len(lines),
           "lines_vertical": int((np.abs(ang - 90.0) <= 15.0).sum()),
           "lines_oblique": int(((np.minimum(ang, 180.0 - ang) >= 20.0) & (np.minimum(ang, 180.0 - ang) <= 70.0)).sum())}
    if ref is not None:
        ref_edges = canny_edges(ref)
        out["edge_f1"] = edge_f1(edges, ref_edges)[2]
        out["edge_f1_auto"] = edge_f1(canny_edges(img, True), canny_edges(ref, True))[2]
        out["line_recall"] = line_recall(lines, hough_lines(ref_edges))
    return out
