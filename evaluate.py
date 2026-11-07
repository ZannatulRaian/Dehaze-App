"""Evaluation of the Dehaze App.

    python evaluate.py                  # writes everything to ./results (about 3 minutes)
    python evaluate.py --export-urban   # also (re)writes the synthetic city test set to assets/
    python evaluate.py --fit-strength   # also re-runs the fit of the haze-adaptive strength rule (adds ~2 minutes)
    python get_street_photos.py         # (once) downloads the real street photos used in section 2

Experiments, all measured against the clear photo behind each scene:

  1. REAL fog      - the 7 hazy photos in assets/test images/outdoor_fog_with_clear_reference (gardens, parks), plus any
                     real street pairs you add to assets/test images/urban_with_clear_reference (see README).
  2. REAL STREETS  - 9 clear photos of real city streets (get_street_photos.py), fogged with the scattering model
                     I = J*t + A*(1-t) and a rough depth guess. Adds a vehicle and pedestrian detection test (detect.py).
  3. SIMULATED     - the 7 clear garden photos, fogged (beta = 1.6) and optionally blurred (Gaussian, horizontal or
                     diagonal motion).
  4. SYNTHETIC     - assets/test images/urban_synthetic: 6 synthetic streets with their true depth map, so the clear scene
                     AND the true transmission are known (checks the dark channel's transmission estimate directly).
  5. HAZE-ADAPTIVE STRENGTH - fog density sweep: fixed default against the strength that follows the measured haze level.
  6. ABLATION      - what each of the three course modules (and the dark channel) adds to the full pipeline.
  7. BLUR ESTIMATION - estimate_blur (type, size, direction) on photos blurred by a known amount, and the Wiener deblur
                     run with the estimated blur instead of the true one.
  8. CROSS-VALIDATION - leave-one-scene-out tuning of the pipeline settings, to see whether the defaults are overfit.
  9. VIDEO         - a synthetic traffic clip with moving cars: per-frame processing against temporal processing.

Tables of 1-4 are followed by a DOWNSTREAM test: what a standard Canny edge detector and probabilistic Hough line detector
find before and after processing, compared with what they find in the clear photo (building edges = near-vertical lines,
lane lines = oblique lines), and in 2 what a YOLOv5n detector finds. Metrics: PSNR (RGB) and SSIM (luminance) against
the clear photo; contrast gain and edge pixels (no reference); Hautiere's visible-edge gain e and gradient ratio r,
which compare a result with its foggy input (no clear photo needed); seconds per image.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import dehaze_core as C
import detect
from dehaze_video import TemporalDehazer, flicker

ROOT = Path(__file__).parent
TEST = ROOT / "assets" / "test images"
DATA = TEST / "outdoor_fog_with_clear_reference"
URBAN_REAL = TEST / "urban_with_clear_reference"      # optional: your own street photos with clear references, see README
URBAN_STREETS = TEST / "urban_real_clear"             # clear real street photos, downloaded by get_street_photos.py
URBAN_SYN = TEST / "urban_synthetic"
OUT = ROOT / "results"
URBAN_BETA = 1.6          # fog density of the hazy images stored in urban_synthetic
BETA = 1.6                # fog density used for the simulated conditions

OFF = dict(C.P_BALANCED, use_dcp=False, hist=0, freq=0, smooth=0, sharpen=0, balance=0)
auto_strength = lambda img: C.auto_strength_params(img)[0]
all_auto = lambda img: C.auto_deblur_params(img, C.auto_strength_params(img)[0])[0]
METHODS = {   # name -> settings (strengths of the single modules are set to a medium 50), or a function image -> settings
    "Foggy input (no processing)": OFF,
    "Histogram stretching only": dict(OFF, hist=50),
    "Frequency-domain enhancement only": dict(OFF, freq=50, cutoff=60),
    "Adaptive smoothing + sharpening only": dict(OFF, smooth=50, sharpen=50),
    "The three course modules together": C.P_MODULES,
    "Dark channel prior only": dict(OFF, use_dcp=True, omega=C.P_BALANCED["omega"], t0=C.P_BALANCED["t0"]),
    "Full pipeline (Balanced default)": C.P_BALANCED,
    "Full pipeline, strength set from the haze level": auto_strength,
    "Traffic monitoring preset (fog removal + histogram stretching)": C.P_TRAFFIC,
    "Traffic monitoring preset, strength set from the haze level": lambda img: C.auto_strength_params(img, C.P_TRAFFIC)[0],
}
INPUT, FULL = "Foggy input (no processing)", "Full pipeline (Balanced default)"
AUTOSTR = "Full pipeline, strength set from the haze level"
KNOWN, AUTO = "Full pipeline + Wiener deblur (blur known)", "Full pipeline + Wiener deblur (blur estimated)"
ALLAUTO = "Full pipeline, haze level and blur both estimated"
ABLATION = {
    "Full pipeline (Balanced default)": C.P_BALANCED,
    "without histogram stretching": dict(C.P_BALANCED, hist=0),
    "without frequency-domain enhancement": dict(C.P_BALANCED, freq=0),
    "without adaptive smoothing + sharpening": dict(C.P_BALANCED, smooth=0, sharpen=0),
    "without all three course modules": dict(C.P_BALANCED, hist=0, freq=0, smooth=0, sharpen=0),
    "without the dark channel prior": dict(C.P_BALANCED, use_dcp=False),
    "without colour balance": dict(C.P_BALANCED, balance=0),
    "without frequency, sharpening, smoothing and colour balance (Traffic preset)": C.P_TRAFFIC,
}


def blur_methods(kind, amount=2.0, angle=0.0):
    """The methods plus the deblur variants: told the true blur, estimating it itself, and everything automatic."""
    return dict(METHODS, **{KNOWN: dict(C.P_BALANCED, deblur=amount, deblur_kind=kind, deblur_angle=angle),
                            AUTO: lambda img: C.auto_deblur_params(img, C.P_BALANCED)[0],
                            ALLAUTO: all_auto})


# ------------------------------------------------------------------ data
def _resize(img, max_dim):
    s = min(1.0, max_dim / max(img.shape[:2]))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def load_pairs(max_dim, folder=DATA):
    """{scene: (hazy, clear)} for every *_hazy.* photo in folder that has a *_clear_reference.* partner."""
    pairs = {}
    for hazy in sorted(p for p in folder.glob("*_hazy.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
        name = hazy.stem.replace("_hazy", "")
        ref = next(iter(folder.glob(f"{name}_clear_reference.*")), None)
        if ref is None:
            continue
        h = cv2.cvtColor(cv2.imread(str(hazy)), cv2.COLOR_BGR2RGB)
        c = cv2.cvtColor(cv2.imread(str(ref)), cv2.COLOR_BGR2RGB)
        s = min(1.0, max_dim / max(h.shape[:2]))
        h = cv2.resize(h, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        c = cv2.resize(c, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_AREA)
        pairs[name] = (h, c)
    return pairs


def load_streets(max_dim):
    """{scene: clear image} of the real street photos fetched by get_street_photos.py (empty if not downloaded)."""
    if not URBAN_STREETS.exists():
        return {}
    return {p.stem.split("_")[0]: _resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), max_dim)
            for p in sorted(URBAN_STREETS.glob("*.jpg"))}


def export_urban(n=6):
    """Write the synthetic city test set: cityNN_clear_reference.png, cityNN_hazy.png (true depth, beta 1.6)
    and cityNN_depth.png (16-bit, 0 = near, 1 = far)."""
    URBAN_SYN.mkdir(parents=True, exist_ok=True)
    for i, (img, depth) in enumerate(C.make_urban_set(n)):
        stem = URBAN_SYN / f"city{i + 1:02d}"
        for suffix, im in (("_clear_reference", img), ("_hazy", C.simulate_fog(img, depth, URBAN_BETA))):
            cv2.imwrite(f"{stem}{suffix}.png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{stem}_depth.png", np.round(depth * 65535).astype(np.uint16))


def load_urban():
    """{scene: (clear, depth)} read from the synthetic test set on disk (written first if it is missing)."""
    if not list(URBAN_SYN.glob("city*_depth.png")):
        export_urban()
    out = {}
    for d in sorted(URBAN_SYN.glob("city*_depth.png")):
        name = d.stem.replace("_depth", "")
        clear = cv2.cvtColor(cv2.imread(str(URBAN_SYN / f"{name}_clear_reference.png")), cv2.COLOR_BGR2RGB)
        out[name] = (clear, cv2.imread(str(d), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0)
    return out


# ------------------------------------------------------------------ running and summarising
def run(methods, scenes, detection=False):
    """scenes: {name: (input, clear)} -> (result rows, {(scene, method): output image}).
    A method is a settings dict, or a function image -> settings (used for the automatic estimates, whose time
    is included in the seconds). detection=True also runs the vehicle / pedestrian detector on every result."""
    rows, outputs, ref_boxes = [], {}, {}
    for scene, (inp, clear) in scenes.items():
        base = C.quality_metrics(inp)
        if detection:
            ref_boxes[scene] = detect.detect(clear)
        for method, P in methods.items():
            t = time.perf_counter()
            out = C.dehaze_pipeline(inp, P(inp) if callable(P) else P)
            sec = time.perf_counter() - t
            q, vis = C.quality_metrics(out), C.visibility_metrics(inp, out)
            row = dict(scene=scene, method=method, psnr=C.psnr(out, clear), ssim=C.ssim(out, clear),
                       contrast_x=q["contrast"] / base["contrast"], edges_pct=q["edges"], vis_e=vis["e"], vis_r=vis["r"],
                       vis_clip=vis["sigma"], seconds=sec, **C.downstream_metrics(out, clear))
            if detection:
                boxes = detect.detect(out)
                rec, prec, _ = detect.match(boxes, ref_boxes[scene])
                row.update(det_found=len(boxes), det_ref=len(ref_boxes[scene]), det_recall=rec, det_precision=prec)
            rows.append(row)
            outputs[(scene, method)] = out
    return rows, outputs


def mean_of(rows, method, key):
    return float(np.nanmean([r[key] for r in rows if r["method"] == method]))


def summarise(rows):
    methods = list(dict.fromkeys(r["method"] for r in rows))
    return [(m, *[mean_of(rows, m, k) for k in ("psnr", "ssim", "contrast_x", "edges_pct", "vis_e", "vis_r", "seconds")])
            for m in methods]


def md_table(summary):
    lines = ["| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m, p, s, c, e, ve, vr, t in summary:
        lines.append(f"| {m} | {p:.1f} | {s:.3f} | {c:.2f} | {e:.1f} | {ve:+.2f} | {vr:.2f} | {t:.2f} |")
    return "\n".join(lines)


def downstream_table(rows):
    """Mean downstream-test results per method, and in how many scenes edge F1 beats the unprocessed input."""
    methods = list(dict.fromkeys(r["method"] for r in rows))
    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    det = "det_recall" in rows[0]
    base = {r["scene"]: r["edge_f1"] for r in rows if r["method"] == INPUT}
    head = ("| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) "
            "| Oblique (lane) | Line recall vs clear | F1 above input |" + (" Vehicles/people found | Detection recall vs clear | Detection precision |" if det else ""))
    lines = [head, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" + (" ---: | ---: | ---: |" if det else "")]
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        mean = lambda k: float(np.nanmean([r[k] for r in sel]))
        better = "-" if m == INPUT else f"{sum(r['edge_f1'] > base[r['scene']] + 1e-9 for r in sel)}/{len(scenes)}"
        line = (f"| {m} | {mean('canny_pct'):.1f} | {mean('edge_f1'):.3f} | {mean('edge_f1_auto'):.3f} | {mean('lines'):.0f} | "
                f"{mean('lines_vertical'):.0f} | {mean('lines_oblique'):.0f} | {mean('line_recall'):.3f} | {better} |")
        if det:
            line += f" {mean('det_found'):.1f} (clear photo: {mean('det_ref'):.1f}) | {mean('det_recall'):.3f} | {mean('det_precision'):.3f} |"
        lines.append(line)
    return "\n".join(lines)


def section(title, rows, md, level="##"):
    md += [f"{level} {title}", "", md_table(summarise(rows)), "", "Downstream test (Canny + Hough" +
           (" + vehicle/pedestrian detection" if "det_recall" in rows[0] else "") + ", means per image):", "",
           downstream_table(rows), ""]


def write_csv(path, rows):
    names = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def montage(images, labels, path, width=360):
    tiles = []
    for img, lab in zip(images, labels):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        t = cv2.resize(img, (width, int(width * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        bar = np.full((28, width, 3), 255, np.uint8)
        cv2.putText(bar, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, t]))
    h = min(t.shape[0] for t in tiles)
    cv2.imwrite(str(path), cv2.cvtColor(np.hstack([t[:h] for t in tiles]), cv2.COLOR_RGB2BGR))


def lines_overlay(img):
    """The image with its Hough segments drawn on top (red)."""
    out = img.copy()
    for x1, y1, x2, y2 in C.hough_lines(C.canny_edges(img)):
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 40, 40), 1, cv2.LINE_AA)
    return out


def boxes_overlay(img):
    """The image with the detector's boxes drawn on top (green vehicles, yellow people)."""
    out = img.copy()
    for x1, y1, x2, y2, sc, name in detect.detect(img):
        col = (255, 220, 0) if name == "person" else (30, 220, 30)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
    return out


# ------------------------------------------------------------------ transmission check (needs true depth)
def transmission_rows(urban):
    """How well the dark channel's transmission map matches the true one exp(-beta * depth)."""
    rows = []
    t0 = C.P_BALANCED["t0"]
    for beta in (0.8, 1.6, 2.4, 3.2):
        for name, (clear, depth) in urban.items():
            hazy = C.simulate_fog(clear, depth, beta)
            t_est, A = C.estimate_transmission(hazy, C.P_BALANCED["omega"], t0, C.P_BALANCED["patch"])
            t_true = np.clip(C.true_transmission(depth, beta), t0, 1.0)
            rows.append(dict(scene=name, beta=beta, t_mae=float(np.abs(t_est - t_true).mean()),
                             t_corr=float(np.corrcoef(t_est.ravel(), t_true.ravel())[0, 1]),
                             airlight_err=float(np.abs(A * 255.0 - C.FOG_AIRLIGHT).mean())))
    return rows


# ------------------------------------------------------------------ fog density sweep, fixed vs haze-adaptive strength
def sweep_table(datasets, betas=(0.4, 0.8, 1.6, 2.4, 3.2)):
    """datasets: {label: {scene: (clear, depth or None)}}. Fixed Balanced default against the strength set from the
    measured haze level, per fog density."""
    methods = {INPUT: OFF, FULL: C.P_BALANCED, AUTOSTR: auto_strength}
    lines = ["| Scenes | Fog beta | Haze level | Strength used | Input SSIM | Fixed SSIM | Adaptive SSIM | Fixed PSNR | Adaptive PSNR "
             "| Fixed edge F1 | Adaptive edge F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows_all = []
    for label, ds in datasets.items():
        for beta in betas:
            scenes = {n: (C.simulate_fog(c, d, beta), c) for n, (c, d) in ds.items()}
            rows, _ = run(methods, scenes)
            hz = [C.haze_level(inp) for inp, _ in scenes.values()]
            g = lambda m, k: mean_of(rows, m, k)
            lines.append(f"| {label} | {beta:g} | {np.mean(hz):.2f} | {np.mean([C.strength_scale(h) for h in hz]):.2f} | "
                         f"{g(INPUT, 'ssim'):.3f} | {g(FULL, 'ssim'):.3f} | {g(AUTOSTR, 'ssim'):.3f} | {g(FULL, 'psnr'):.1f} | "
                         f"{g(AUTOSTR, 'psnr'):.1f} | {g(FULL, 'edge_f1'):.3f} | {g(AUTOSTR, 'edge_f1'):.3f} |")
            rows_all += [dict(r, dataset=label, beta=beta) for r in rows]
    return rows_all, "\n".join(lines)


def fit_strength_rule(datasets, betas=(0.4, 0.8, 1.2, 1.6, 2.4, 3.2), gs=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25)):
    """Re-derive strength = clip(a * haze - b, 0.25, 1) and check it on held-out scene types. For every scene and fog
    density, the Balanced settings are scaled by g (strengths of the four enhancement stages and omega) and scored by
    SSIM + PSNR/40; the rule is fit on two scene types and tested on the third."""
    def scaled(g):
        P = dict(C.P_BALANCED)
        for k in ("hist", "freq", "smooth", "sharpen"):
            P[k] = C.P_BALANCED[k] * g
        P["omega"], P["use_dcp"] = (min(0.95, C.P_BALANCED["omega"] * (0.35 + 0.65 * g)), True) if g > 0 else (0.0, False)
        return P
    recs = []
    for label, ds in datasets.items():
        for beta in betas:
            for n, (c, d) in ds.items():
                f = C.simulate_fog(c, d, beta)
                sc = [(C.ssim(o, c) + C.psnr(o, c) / 40.0) for o in (C.dehaze_pipeline(f, scaled(g)) if g > 0 else f for g in gs)]
                recs.append((label, beta, C.haze_level(f), np.array(sc)))
    score = lambda r, g: float(np.interp(g, gs, r[3]))
    rule = lambda h, a, b: float(np.clip(a * h - b, C.STRENGTH_MIN, 1.0))
    grid = [(a, b) for a in np.arange(1.0, 3.01, 0.1) for b in np.arange(0.0, 1.01, 0.05)]
    fit = lambda tr: max(grid, key=lambda p: np.mean([score(r, rule(r[2], *p)) for r in tr]))
    a, b = fit(recs)
    lines = [f"Fit on all scene types: strength = clip({a:.1f} x haze - {b:.2f}, {C.STRENGTH_MIN}, 1) "
             f"(in dehaze_core: {C.HAZE_SLOPE} and {C.HAZE_OFFSET}).", "",
             "| Held-out scene type | Rule fit on the others | Score, fixed default | Score, adaptive | Score, per-image best | "
             "Light fog only (beta <= 0.8): fixed | adaptive |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for held in datasets:
        tr, te = [r for r in recs if r[0] != held], [r for r in recs if r[0] == held]
        pa, pb = fit(tr)
        lt = [r for r in te if r[1] <= 0.8]
        lines.append(f"| {held} | {pa:.1f} x haze - {pb:.2f} | {np.mean([score(r, 1.0) for r in te]):.3f} | "
                     f"{np.mean([score(r, rule(r[2], pa, pb)) for r in te]):.3f} | {np.mean([r[3].max() for r in te]):.3f} | "
                     f"{np.mean([score(r, 1.0) for r in lt]):.3f} | {np.mean([score(r, rule(r[2], pa, pb)) for r in lt]):.3f} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ ablation
def ablation_table(datasets, det_labels=()):
    """datasets: {label: {scene: (input, clear)}}; mean SSIM, edge F1 and visible-edge gain per ablated pipeline
    (and vehicle/pedestrian detection recall for the datasets named in det_labels)."""
    res = {label: run(ABLATION, scenes, detection=label in det_labels)[0] for label, scenes in datasets.items()}
    cols = lambda l: ["SSIM", "F1", "e"] + (["detection recall"] if l in det_labels else [])
    lines = ["| Pipeline | " + " | ".join(f"{l}: " + " | ".join(cols(l)) for l in res) + " |",
             "| --- | " + " | ".join(" | ".join("---:" for _ in cols(l)) for l in res) + " |"]
    for m in ABLATION:
        cells = []
        for l, rows in res.items():
            cells += [f"{mean_of(rows, m, 'ssim'):.3f}", f"{mean_of(rows, m, 'edge_f1'):.3f}", f"{mean_of(rows, m, 'vis_e'):+.2f}"]
            if l in det_labels:
                cells.append(f"{mean_of(rows, m, 'det_recall'):.3f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ blur estimation experiment
def _angle_err(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def blur_estimation(real, urban):
    """Estimate the blur of photos blurred by a known amount (fog beta 1.6 unless noted); motion blur at several angles.
    real / urban: {name: clear image}. Returns (csv rows, markdown tables)."""
    angles = (0, 25, 45, 70, 90, 135, 160)
    conds = [("none", 0.0, b) for b in (0.0, 1.6, 3.0)]
    conds += [(k, a, 1.6) for k in ("Gaussian", "Motion") for a in (1.0, 2.0, 3.0, 4.0)]
    conds += [("Gaussian", 2.0, 3.0), ("Motion", 2.0, 3.0)]
    rows = []
    for kind, amount, beta in conds:
        for ds_name, ds in (("real photos", real), ("synthetic cities", urban)):
            for ang in (angles if kind == "Motion" else (0,)):
                for scene, clear in ds.items():
                    img = C.simulate_fog(clear, None, beta) if beta else clear
                    est = C.estimate_blur(C.simulate_blur(img, amount, kind, ang) if amount else img)
                    rows.append(dict(dataset=ds_name, true_kind=kind, true_amount=amount, true_angle=ang if kind == "Motion" else "",
                                     fog_beta=beta, scene=scene, est_kind=est["kind"], est_amount=est["amount"],
                                     est_angle=est["angle"] if est["kind"] == "Motion" else ""))
    lines = ["| True blur | Fog beta | Real: estimate (px) | kind right | abs. error (px) | angle error (deg) "
             "| Synthetic: estimate (px) | kind right | abs. error (px) | angle error (deg) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind, amount, beta in conds:
        cells = []
        for ds_name in ("real photos", "synthetic cities"):
            sel = [r for r in rows if r["dataset"] == ds_name and r["true_kind"] == kind
                   and r["true_amount"] == amount and r["fog_beta"] == beta]
            right = np.mean([r["est_kind"] == kind for r in sel])
            err = np.mean([abs(r["est_amount"] - amount) for r in sel])
            ang = [_angle_err(r["est_angle"], r["true_angle"]) for r in sel if kind == "Motion" and r["est_kind"] == "Motion"]
            cells += [f"{np.mean([r['est_amount'] for r in sel]):.2f}", f"{right * 100:.0f}%", f"{err:.2f}",
                      f"{np.mean(ang):.1f}" if ang else "-"]
        label = "none (sharp)" if kind == "none" else f"{kind} {amount:g} px" + (" (7 angles)" if kind == "Motion" else "")
        lines.append(f"| {label} | {beta:g} | " + " | ".join(cells) + " |")
    ang_lines = ["| True motion angle | Real: mean angle error (deg) | Real: estimate at 3 px | Synthetic: mean angle error (deg) | Synthetic: estimate at 3 px |",
                 "| ---: | ---: | ---: | ---: | ---: |"]
    for ang in angles:
        cells = []
        for ds_name in ("real photos", "synthetic cities"):
            sel = [r for r in rows if r["dataset"] == ds_name and r["true_kind"] == "Motion" and r["true_angle"] == ang
                   and r["true_amount"] >= 2.0 and r["fog_beta"] == 1.6 and r["est_kind"] == "Motion"]
            s3 = [r["est_amount"] for r in sel if r["true_amount"] == 3.0]
            cells += [f"{np.mean([_angle_err(r['est_angle'], ang) for r in sel]):.1f}", f"{np.mean(s3):.2f}"]
        ang_lines.append(f"| {ang} | " + " | ".join(cells) + " |")
    return rows, "\n".join(lines), "\n".join(ang_lines)


# ------------------------------------------------------------------ cross-validation of the settings
def cross_validate(pairs, n=60, seed=0, max_dim=400):
    """Leave-one-scene-out tuning. n random settings (strengths, omega, colour balance) are scored on every real fog
    scene (SSIM + PSNR/40); for each scene the best setting on the OTHER scenes is applied to it. Compares that with the
    shipped default (which was tuned on these photos, so its number is optimistic) and with the best setting per scene."""
    rng = np.random.default_rng(seed)
    space = dict(hist=(0, 60), freq=(0, 60), smooth=(0, 50), sharpen=(0, 80), omega=(0.6, 0.95), balance=(0, 100))
    cands = [dict(C.P_BALANCED, **{k: float(rng.uniform(*v)) for k, v in space.items()}) for _ in range(n)]
    small = {k: (_resize(h, max_dim), _resize(c, max_dim)) for k, (h, c) in pairs.items()}
    sc = lambda P: np.array([[C.ssim(o, c), C.psnr(o, c)] for o, c in
                             ((C.dehaze_pipeline(h, P), c) for h, c in small.values())])
    S = np.stack([sc(P) for P in cands])                  # (candidate, scene, [ssim, psnr])
    D = sc(C.P_BALANCED)
    comb = S[..., 0] + S[..., 1] / 40.0
    held = []
    for i in range(len(small)):
        others = [j for j in range(len(small)) if j != i]
        best = int(comb[:, others].mean(axis=1).argmax())
        held.append(S[best, i])
    held = np.array(held)
    oracle = S[comb.argmax(axis=0), np.arange(len(small))]
    return "\n".join([
        f"| Settings | SSIM | PSNR (dB) |", "| --- | ---: | ---: |",
        f"| Shipped default (tuned on these photos) | {D[:, 0].mean():.3f} | {D[:, 1].mean():.2f} |",
        f"| Tuned on the other {len(small) - 1} scenes, applied to the held-out one ({n} random candidates) | {held[:, 0].mean():.3f} | {held[:, 1].mean():.2f} |",
        f"| Best of the {n} candidates for each scene separately (upper bound) | {oracle[:, 0].mean():.3f} | {oracle[:, 1].mean():.2f} |",
        f"| Unprocessed | {np.mean([C.ssim(h, c) for h, c in small.values()]):.3f} | {np.mean([C.psnr(h, c) for h, c in small.values()]):.2f} |"])


# ------------------------------------------------------------------ video
def video_table(n=30, noise=2.0, seed=0):
    """Synthetic traffic clip (fixed camera, moving cars, fog from the true depth, sensor noise): independent frames
    against temporal processing."""
    clear, depth = C.make_traffic_clip(n)
    rng = np.random.default_rng(seed)
    fog = [np.clip(C.simulate_fog(f, depth, BETA).astype(np.float32) + rng.normal(0, noise, f.shape), 0, 255).astype(np.uint8)
           for f in clear]
    lines = ["| Processing | Flicker (grey levels) | PSNR (dB) | SSIM | Seconds/frame |", "| --- | ---: | ---: | ---: | ---: |",
             f"| Foggy input | {flicker(fog, fog):.2f} | {np.mean([C.psnr(o, c) for o, c in zip(fog, clear)]):.1f} | "
             f"{np.mean([C.ssim(o, c) for o, c in zip(fog, clear)]):.3f} | - |"]
    outs = {}
    for name, kw in (("Every frame on its own", dict(smoothing=0.0)), ("Temporal (smoothed estimates + denoising)", dict()),
                     ("Every frame on its own, strength from haze level", dict(smoothing=0.0, auto=True)),
                     ("Temporal, strength from haze level", dict(auto=True))):
        d = TemporalDehazer(**kw)
        t = time.perf_counter()
        out = [d.process(f) for f in fog]
        sec = (time.perf_counter() - t) / n
        outs[name] = out
        lines.append(f"| {name} | {flicker(fog, out):.2f} | {np.mean([C.psnr(o, c) for o, c in zip(out, clear)]):.1f} | "
                     f"{np.mean([C.ssim(o, c) for o, c in zip(out, clear)]):.3f} | {sec:.2f} |")
    k = n // 2
    montage([fog[k], outs["Every frame on its own"][k], outs["Temporal (smoothed estimates + denoising)"][k], clear[k]],
            ["foggy frame", "frame on its own", "temporal", "clear frame"], OUT / "figures" / "video_frame.png")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dim", type=int, default=800, help="downscale test photos to this long edge")
    ap.add_argument("--export-urban", action="store_true", help="(re)write the synthetic city test set to assets/")
    ap.add_argument("--fit-strength", action="store_true", help="re-run the fit of the haze-adaptive strength rule")
    args = ap.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    if args.export_urban:
        export_urban()
    pairs = load_pairs(args.max_dim)
    urban = load_urban()
    streets = load_streets(args.max_dim)
    det_ok = bool(streets) and detect.available()
    md = ["# Evaluation results", "",
          f"{len(pairs)} real garden/park scenes with a clear reference, {len(streets)} real street photos fogged by simulation, "
          f"{len(urban)} synthetic streets with true depth; photos scaled to at most {args.max_dim}px on the long edge. "
          "PSNR and SSIM are measured against the clear photo; the contrast gain is relative to the foggy input and edge pixels "
          "are the share of pixels with a gradient above 12. *Visible-edge gain e* and *gradient ratio r* (Hautiere et al.) "
          "compare a result with its foggy input and need no clear photo: e is the relative gain in visible edges, r the mean "
          "gradient ratio on them (above 1 = contrast gained). Each value is the mean over the scenes. The downstream test runs "
          "a standard Canny edge detector (fixed thresholds 50/150, which is what makes fog costly) and a probabilistic Hough line "
          "detector on every result: *Edge F1* compares its edges with the clear photo's (2 px tolerance); *auto thresholds* repeats "
          "that with Canny thresholds set from each image's median brightness, which compensates for low contrast and so shows the "
          "gain that is not just a brighter or higher-contrast picture; *line recall* is the share of the clear photo's lines found "
          "again; *vertical* lines are within 15 degrees of vertical (building edges) and *oblique* lines are 20-70 degrees from "
          "horizontal (lane and roof lines). More edges or lines is not better by itself: noise and over-sharpening add false ones, "
          "and the Hough line count also depends on how segments merge, so compare with F1 and recall.", ""]

    # 1) real fog
    rows, outs = run(METHODS, pairs)
    write_csv(OUT / "real_fog.csv", rows)
    real_rows = rows
    section("1. Real fog (7 garden and park scenes)", rows, md)
    s = "scene12" if "scene12" in pairs else next(iter(pairs))
    montage([pairs[s][0], outs[(s, "The three course modules together")], outs[(s, FULL)], pairs[s][1]],
            ["foggy photo", "three modules only", "full pipeline", "clear reference"], OUT / "figures" / "real_fog_montage.png")
    s = "scene14" if "scene14" in pairs else s            # the most built-up of the real scenes
    montage([C.canny_edges(pairs[s][0]), C.canny_edges(outs[(s, FULL)]), C.canny_edges(pairs[s][1])],
            ["Canny: foggy", "Canny: full pipeline", "Canny: clear"], OUT / "figures" / "downstream_edges_real.png")
    montage([lines_overlay(pairs[s][0]), lines_overlay(outs[(s, FULL)]), lines_overlay(pairs[s][1])],
            ["Hough lines: foggy", "Hough lines: full pipeline", "Hough lines: clear"], OUT / "figures" / "downstream_lines_real.png")

    street_pairs = load_pairs(args.max_dim, URBAN_REAL) if URBAN_REAL.exists() else {}
    if street_pairs:
        rows, _ = run(METHODS, street_pairs)
        write_csv(OUT / "street_pairs.csv", rows)
        section(f"1b. Real city street photos with real fog ({len(street_pairs)} pairs)", rows, md)

    # 2) real street photos, simulated fog
    if streets:
        md += ["## 2. Real city streets, simulated fog", "",
               f"{len(streets)} clear photos of real streets (`python get_street_photos.py`; one is `assets/test images/2467248.jpg`), fogged "
               f"with the scattering model (beta {BETA}) and a rough depth guess (farther toward the top of the frame). "
               + ("*Vehicles/people found* are the detections of a YOLOv5n detector (cars, buses, trucks, motorcycles, bicycles, "
                  "pedestrians); recall and precision treat the detector's output on the clear photo as the reference (IoU 0.5). "
                  if det_ok else "The vehicle detector (detect.py) could not be loaded, so that test was skipped. "), ""]
        scenes = {n: (C.simulate_fog(c, None, BETA), c) for n, c in streets.items()}
        rows, outs = run(METHODS, scenes, detection=det_ok)
        write_csv(OUT / "street_photos.csv", rows)
        section(f"2. Real streets, fog beta {BETA}", rows, md, "###")
        s = next(iter(scenes))
        montage([scenes[s][0], outs[(s, FULL)], scenes[s][1]], ["foggy street", "full pipeline", "clear street"],
                OUT / "figures" / "street_montage.png", width=420)
        if det_ok:
            s = max(scenes, key=lambda n: sum(1 for b in detect.detect(streets[n]) if b[5] in detect.VEHICLES))
            montage([boxes_overlay(scenes[s][0]), boxes_overlay(outs[(s, FULL)]), boxes_overlay(scenes[s][1])],
                    ["detections: foggy", "detections: full pipeline", "detections: clear"], OUT / "figures" / "detection_street.png", width=420)

    # 3) simulated fog and blur on the clear garden photos
    conditions = {
        "fog": (lambda c: C.simulate_fog(c, None, BETA), None, 0.0),
        "fog + Gaussian blur (2 px)": (lambda c: C.simulate_blur(C.simulate_fog(c, None, BETA), 2.0, "Gaussian"), "Gaussian", 0.0),
        "fog + horizontal motion blur (2 px)": (lambda c: C.simulate_blur(C.simulate_fog(c, None, BETA), 2.0, "Motion"), "Motion", 0.0),
        "fog + diagonal motion blur (45 deg, 2 px)": (lambda c: C.simulate_blur(C.simulate_fog(c, None, BETA), 2.0, "Motion", 45.0), "Motion", 45.0),
    }
    all_rows = []
    for cond, (make, kind, ang) in conditions.items():
        scenes = {n: (make(c), c) for n, (_, c) in pairs.items()}
        rows, outs = run(blur_methods(kind, 2.0, ang) if kind else dict(METHODS), scenes)
        all_rows += [dict(r, condition=cond) for r in rows]
        section(f"3. Simulated: {cond}", rows, md)
        if "diagonal" in cond:
            s = next(iter(scenes))
            montage([scenes[s][0], outs[(s, FULL)], outs[(s, AUTO)], scenes[s][1]],
                    ["fog + diagonal motion blur", "full pipeline", "+ Wiener (blur estimated)", "clear photo"],
                    OUT / "figures" / "simulated_motion_blur_montage.png")
    write_csv(OUT / "simulated.csv", all_rows)

    # 4) synthetic urban set with true depth
    md += ["## 4. Synthetic city streets with true depth", "",
           f"{len(urban)} synthetic streets (assets/test images/urban_synthetic, 640x400, different building layouts, "
           "cars, lane lines), fogged with t = exp(-beta * depth) from their true depth map.", ""]
    urban_rows, trans = [], transmission_rows(urban)
    for i, (cond, blur, ang) in enumerate((("fog", None, 0.0), ("fog + horizontal motion blur (2 px)", "Motion", 0.0),
                                           ("fog + diagonal motion blur (45 deg, 2 px)", "Motion", 45.0),
                                           ("fog + Gaussian blur (2 px)", "Gaussian", 0.0)), 1):
        scenes = {}
        for n, (c, d) in urban.items():
            f = C.simulate_fog(c, d, BETA)
            scenes[n] = (C.simulate_blur(f, 2.0, blur, ang) if blur else f, c)
        rows, outs = run(blur_methods(blur, 2.0, ang) if blur else dict(METHODS), scenes)
        urban_rows += [dict(r, condition=cond) for r in rows]
        section(f"4.{i}. Synthetic streets: {cond} (beta {BETA})", rows, md, "###")
        if cond == "fog":
            s = next(iter(scenes))
            montage([scenes[s][0], outs[(s, "Dark channel prior only")], outs[(s, FULL)], scenes[s][1]],
                    ["foggy city", "dark channel only", "full pipeline", "clear city"], OUT / "figures" / "urban_montage.png")
            montage([lines_overlay(scenes[s][0]), lines_overlay(outs[(s, FULL)]), lines_overlay(scenes[s][1])],
                    ["Hough lines: foggy", "Hough lines: full pipeline", "Hough lines: clear"], OUT / "figures" / "downstream_lines_urban.png")
    write_csv(OUT / "urban.csv", urban_rows)
    md += ["### Transmission estimate against the true depth", "",
           "Mean over the synthetic set; the true transmission is exp(-beta * depth), clipped to the same minimum as the estimate.", "",
           "| Beta | Transmission MAE | Correlation with true t | Airlight error (0-255) |", "| ---: | ---: | ---: | ---: |"]
    for beta in (0.8, 1.6, 2.4, 3.2):
        sel = [r for r in trans if r["beta"] == beta]
        md.append(f"| {beta:g} | {np.mean([r['t_mae'] for r in sel]):.3f} | {np.mean([r['t_corr'] for r in sel]):.3f} "
                  f"| {np.mean([r['airlight_err'] for r in sel]):.1f} |")
    md.append("")
    write_csv(OUT / "transmission.csv", trans)

    # 5) haze-adaptive strength
    datasets = {"7 garden photos": {n: (c, None) for n, (_, c) in pairs.items()}, "synthetic streets": dict(urban)}
    if streets:
        datasets["9 real streets"] = {n: (c, None) for n, c in streets.items()}
    sweep_rows, sweep_md = sweep_table(datasets)
    write_csv(OUT / "strength_sweep.csv", sweep_rows)
    md += ["## 5. Fog density sweep: fixed default against strength set from the haze level", "",
           "Fog is simulated at each density (garden photos and street photos with the rough depth guess, synthetic streets with their "
           "true depth). *Haze level* is `haze_level()` of the foggy image; *strength used* is the share of the default's strengths "
           "that `auto_strength_params` applies. The default's strengths are tuned for dense fog; on light haze they over-process.", "",
           sweep_md, ""]
    g = lambda m, k: mean_of(real_rows, m, k)
    md += ["On the 7 **real** fog photos (section 1; real fog was not used for the fit), fixed against adaptive default: "
           f"PSNR {g(FULL, 'psnr'):.2f} / {g(AUTOSTR, 'psnr'):.2f} dB, SSIM {g(FULL, 'ssim'):.3f} / {g(AUTOSTR, 'ssim'):.3f}, "
           f"edge F1 {g(FULL, 'edge_f1'):.3f} / {g(AUTOSTR, 'edge_f1'):.3f}.", ""]
    if args.fit_strength:
        fit_md = fit_strength_rule(datasets)
        (OUT / "strength_fit.md").write_text(fit_md, encoding="utf-8")
    if (OUT / "strength_fit.md").exists():
        md += ["### Fit of the rule and held-out check", "",
               "The rule was fit on simulated fog of two scene types and tested on the third (python evaluate.py --fit-strength). "
               "Score is SSIM + PSNR/40; per-image best is the best of six fixed scalings chosen with knowledge of the clear photo.", "",
               (OUT / "strength_fit.md").read_text(encoding="utf-8"), ""]

    # 6) ablation
    abl_sets = {"Real fog": {n: (h, c) for n, (h, c) in pairs.items()},
                "Synthetic streets": {n: (C.simulate_fog(c, d, BETA), c) for n, (c, d) in urban.items()}}
    if streets:
        abl_sets["Real streets"] = {n: (C.simulate_fog(c, None, BETA), c) for n, c in streets.items()}
    md += ["## 6. What each module adds to the full pipeline", "",
           "Each row removes one part of the Balanced pipeline; compare with the first row. SSIM and edge F1 are against the clear photo, "
           "e is the visible-edge gain over the foggy input; detection recall is the share of the clear street photos' vehicles and people "
           "that the YOLOv5n detector still finds.", "", ablation_table(abl_sets, ("Real streets",) if det_ok else ()), ""]

    # 7) blur estimation
    est_rows, est_md, ang_md = blur_estimation({n: c for n, (_, c) in pairs.items()}, {n: c for n, (c, _) in urban.items()})
    write_csv(OUT / "blur_estimation.csv", est_rows)
    hazy_est = [C.estimate_blur(h) for h, _ in pairs.values()]
    md += ["## 7. Blur estimation", "",
           "`estimate_blur` measures the width of the strongest edges along 8 directions and subtracts the width of a sharp photo. "
           "Each row blurs the clear photos by a known amount (fog added first, beta 1.6 unless the row says otherwise) and compares the "
           "estimate with it; motion blur rows pool 7 angles (0, 25, 45, 70, 90, 135, 160 degrees), length 3 x amount. The calibration "
           "factors were fit on the real photos, so the synthetic cities are an independent check.", "", est_md, "",
           "Motion angle, by true angle (blur 2 to 4 px, fog beta 1.6):", "", ang_md, "",
           f"On the {len(pairs)} real hazy photos (true blur unknown) the estimator found: "
           f"{', '.join(f'{k} {sum(e['kind'] == k for e in hazy_est)}' for k in dict.fromkeys(e['kind'] for e in hazy_est))} "
           f"(largest estimate {max(e['amount'] for e in hazy_est):.2f} px).", ""]

    # 8) cross-validation
    md += ["## 8. Are the defaults overfit? Leave-one-scene-out tuning (real fog)", "",
           "60 random settings (strengths, dark channel omega, colour balance) are scored on the 7 real fog scenes at 400 px. "
           "For each scene, the best setting on the other six is applied to it. The shipped default was tuned on a subset of these "
           "photos, so its row is optimistic; the street and synthetic results above are data it never saw.", "",
           cross_validate(pairs), ""]

    # 9) video
    md += ["## 9. Video: synthetic traffic clip", "",
           "30 frames, 640x400, fixed camera, 3 cars driving across, fog (beta 1.6) from the true depth and sensor noise (sigma 2). "
           "*Flicker* is the mean frame-to-frame change of the output in pixels that did not change in the input (static background); "
           "the clip is processed by `dehaze_video.TemporalDehazer`.", "", video_table(), ""]

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
