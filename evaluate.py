"""Evaluation of the Dehaze App on the bundled test scenes.

    python evaluate.py                  # writes everything to ./results
    python evaluate.py --export-urban   # also (re)writes the synthetic city test set to assets/

Experiments, all measured against the clear reference of each scene:

  1. REAL fog     - the 7 hazy photos in assets/test images/outdoor_fog_with_clear_reference
                    (plus any pairs you add to assets/test images/urban_with_clear_reference, see README).
  2. SIMULATED    - the 7 clear photos, fogged with the scattering model I = J*t + A*(1-t)
                    (beta = 1.6), optionally also blurred (Gaussian or horizontal motion).
  3. URBAN        - assets/test images/urban_synthetic: 6 synthetic city streets with their true depth map.
                    The fog is simulated from that depth, so the clear scene AND the true transmission are
                    known (checks the dark channel's transmission estimate directly).
  4. BLUR ESTIMATION - dehaze_core.estimate_blur on photos blurred by a known amount, and the Wiener deblur
                    run with the estimated blur instead of the true one (also a column in the tables of 2 and 3).

Every table of 1-3 is followed by a DOWNSTREAM test: what a standard Canny edge detector and probabilistic
Hough line detector find before and after processing, compared with what they find in the clear photo
(building edges = near-vertical lines, lane lines = oblique lines).

Each method is an ablation of the pipeline, so the contribution of the three course modules
(histogram stretching, frequency-domain enhancement, adaptive smoothing + sharpening) shows up
on its own, then together, then combined with the dark channel prior.

Metrics: PSNR (RGB) and SSIM (luminance) against the clear photo; contrast gain relative to the
foggy input and the share of edge pixels (no reference needed); seconds per image.
Only numpy / OpenCV / SciPy are used, no Streamlit.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import dehaze_core as C

ROOT = Path(__file__).parent
TEST = ROOT / "assets" / "test images"
DATA = TEST / "outdoor_fog_with_clear_reference"
URBAN_REAL = TEST / "urban_with_clear_reference"      # optional: your own street photos, see README
URBAN_SYN = TEST / "urban_synthetic"
OUT = ROOT / "results"
URBAN_BETA = 1.6          # fog density of the hazy images stored in urban_synthetic

OFF = dict(C.P_BALANCED, use_dcp=False, hist=0, freq=0, smooth=0, sharpen=0, balance=0)
METHODS = {   # name -> settings (strengths of the single modules are set to a medium 50)
    "Foggy input (no processing)": OFF,
    "Histogram stretching only": dict(OFF, hist=50),
    "Frequency-domain enhancement only": dict(OFF, freq=50, cutoff=60),
    "Adaptive smoothing + sharpening only": dict(OFF, smooth=50, sharpen=50),
    "The three course modules together": C.P_MODULES,
    "Dark channel prior only": dict(OFF, use_dcp=True, omega=C.P_BALANCED["omega"], t0=C.P_BALANCED["t0"]),
    "Full pipeline (Balanced default)": C.P_BALANCED,
}
INPUT, FULL = "Foggy input (no processing)", "Full pipeline (Balanced default)"
KNOWN, AUTO = "Full pipeline + Wiener deblur (blur known)", "Full pipeline + Wiener deblur (blur estimated)"


def blur_methods(kind, amount=2.0):
    """The methods plus the two deblur variants: told the true blur, and estimating it itself."""
    return dict(METHODS, **{KNOWN: dict(C.P_BALANCED, deblur=amount, deblur_kind=kind),
                            AUTO: lambda img: C.auto_deblur_params(img, C.P_BALANCED)[0]})


# ------------------------------------------------------------------ data
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
def run(methods, scenes):
    """scenes: {name: (input, clear)} -> (result rows, {(scene, method): output image}).
    A method is a settings dict, or a function image -> settings (used for the automatic blur estimate,
    whose time is included in the seconds)."""
    rows, outputs = [], {}
    for scene, (inp, clear) in scenes.items():
        base = C.quality_metrics(inp)
        for method, P in methods.items():
            t = time.perf_counter()
            out = C.dehaze_pipeline(inp, P(inp) if callable(P) else P)
            sec = time.perf_counter() - t
            q = C.quality_metrics(out)
            rows.append(dict(scene=scene, method=method, psnr=C.psnr(out, clear), ssim=C.ssim(out, clear),
                             contrast_x=q["contrast"] / base["contrast"], edges_pct=q["edges"], seconds=sec,
                             **C.downstream_metrics(out, clear)))
            outputs[(scene, method)] = out
    return rows, outputs


def summarise(rows):
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = []
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        out.append((m, *[float(np.mean([r[k] for r in sel])) for k in ("psnr", "ssim", "contrast_x", "edges_pct", "seconds")]))
    return out


def md_table(summary):
    lines = ["| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for m, p, s, c, e, t in summary:
        lines.append(f"| {m} | {p:.1f} | {s:.3f} | {c:.2f} | {e:.1f} | {t:.2f} |")
    return "\n".join(lines)


def downstream_table(rows):
    """Mean downstream-test results per method, and in how many scenes edge F1 beats the unprocessed input."""
    methods = list(dict.fromkeys(r["method"] for r in rows))
    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    base = {r["scene"]: r["edge_f1"] for r in rows if r["method"] == INPUT}
    lines = ["| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines "
             "| Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        mean = lambda k: float(np.nanmean([r[k] for r in sel]))
        better = "-" if m == INPUT else f"{sum(r['edge_f1'] > base[r['scene']] + 1e-9 for r in sel)}/{len(scenes)}"
        lines.append(f"| {m} | {mean('canny_pct'):.1f} | {mean('edge_f1'):.3f} | {mean('edge_f1_auto'):.3f} | {mean('lines'):.0f} | "
                     f"{mean('lines_vertical'):.0f} | {mean('lines_oblique'):.0f} | {mean('line_recall'):.3f} | {better} |")
    return "\n".join(lines)


def section(title, rows, md, level="##"):
    md += [f"{level} {title}", "", md_table(summarise(rows)), "", "Downstream test (Canny + Hough, means per image):", "",
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


# ------------------------------------------------------------------ blur estimation experiment
def blur_estimation(real, urban):
    """Estimate the blur of photos blurred by a known amount (fog beta 1.6 unless noted).
    real / urban: {name: clear image}. Returns (csv rows, markdown table)."""
    conds = [("none", 0.0, b) for b in (0.0, 1.6, 3.0)]
    conds += [(k, a, 1.6) for k in ("Gaussian", "Motion") for a in (1.0, 2.0, 3.0, 4.0)]
    conds += [("Gaussian", 2.0, 3.0), ("Motion", 2.0, 3.0)]
    rows = []
    for kind, amount, beta in conds:
        for ds_name, ds in (("real photos", real), ("synthetic cities", urban)):
            for scene, clear in ds.items():
                img = C.simulate_fog(clear, None, beta) if beta else clear
                est = C.estimate_blur(C.simulate_blur(img, amount, kind) if amount else img)
                rows.append(dict(dataset=ds_name, true_kind=kind, true_amount=amount, fog_beta=beta, scene=scene,
                                 est_kind=est["kind"], est_amount=est["amount"]))
    lines = ["| True blur | Fog beta | Real photos: estimate (px) | kind right | abs. error (px) "
             "| Synthetic cities: estimate (px) | kind right | abs. error (px) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind, amount, beta in conds:
        cells = []
        for ds_name in ("real photos", "synthetic cities"):
            sel = [r for r in rows if r["dataset"] == ds_name and r["true_kind"] == kind
                   and r["true_amount"] == amount and r["fog_beta"] == beta]
            right = np.mean([r["est_kind"] == kind for r in sel])
            err = np.mean([abs(r["est_amount"] - amount) for r in sel])
            cells += [f"{np.mean([r['est_amount'] for r in sel]):.2f}", f"{right * 100:.0f}%", f"{err:.2f}"]
        label = "none (sharp)" if kind == "none" else f"{kind} {amount:g} px"
        lines.append(f"| {label} | {beta:g} | " + " | ".join(cells) + " |")
    return rows, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dim", type=int, default=800, help="downscale test photos to this long edge")
    ap.add_argument("--export-urban", action="store_true", help="(re)write the synthetic city test set to assets/")
    args = ap.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    if args.export_urban:
        export_urban()
    pairs = load_pairs(args.max_dim)
    urban = load_urban()
    md = ["# Evaluation results", "",
          f"{len(pairs)} real scenes, photos scaled to at most {args.max_dim}px on the long edge. "
          "PSNR and SSIM are measured against the clear reference photo of each scene; the contrast gain is "
          "relative to the foggy input and edge pixels are the share of pixels with a gradient above 12. "
          "Each value is the mean over the scenes. The downstream test runs a standard Canny edge detector "
          "(fixed thresholds 50/150, which is what makes fog costly) and a probabilistic Hough line detector on every "
          "result: *Edge F1* compares its edges with the clear photo's (2 px tolerance); *auto thresholds* repeats that "
          "with Canny thresholds set from each image's median brightness, which compensates for low contrast and so "
          "shows the gain that is not just a brighter or higher-contrast picture; *line recall* is the share of the clear "
          "photo's lines found again; *vertical* lines are within 15 degrees of vertical (building edges) and *oblique* "
          "lines are 20-70 degrees from horizontal (lane and roof lines). More edges or lines is not better by itself: "
          "noise and over-sharpening add false ones, and the Hough line count also depends on how segments merge, so "
          "compare with F1 and recall.", ""]

    # 1) real fog
    rows, outs = run(METHODS, pairs)
    write_csv(OUT / "real_fog.csv", rows)
    section("1. Real fog", rows, md)
    s = "scene12" if "scene12" in pairs else next(iter(pairs))
    montage([pairs[s][0], outs[(s, "The three course modules together")], outs[(s, FULL)], pairs[s][1]],
            ["foggy photo", "three modules only", "full pipeline", "clear reference"], OUT / "figures" / "real_fog_montage.png")
    s = "scene14" if "scene14" in pairs else s            # the most built-up of the real scenes
    montage([C.canny_edges(pairs[s][0]), C.canny_edges(outs[(s, FULL)]), C.canny_edges(pairs[s][1])],
            ["Canny: foggy", "Canny: full pipeline", "Canny: clear"], OUT / "figures" / "downstream_edges_real.png")
    montage([lines_overlay(pairs[s][0]), lines_overlay(outs[(s, FULL)]), lines_overlay(pairs[s][1])],
            ["Hough lines: foggy", "Hough lines: full pipeline", "Hough lines: clear"],
            OUT / "figures" / "downstream_lines_real.png")

    # optional: the user's own street photos with clear references
    street = load_pairs(args.max_dim, URBAN_REAL) if URBAN_REAL.exists() else {}
    if street:
        rows, _ = run(METHODS, street)
        write_csv(OUT / "street_photos.csv", rows)
        section(f"1b. Real city street photos ({len(street)} pairs)", rows, md)

    # 2) simulated fog and blur on the clear photos
    conditions = {
        "fog": lambda c: C.simulate_fog(c, None, 1.6),
        "fog + Gaussian blur (2 px)": lambda c: C.simulate_blur(C.simulate_fog(c, None, 1.6), 2.0, "Gaussian"),
        "fog + horizontal motion blur (2 px)": lambda c: C.simulate_blur(C.simulate_fog(c, None, 1.6), 2.0, "Motion"),
    }
    all_rows = []
    for cond, make in conditions.items():
        scenes = {n: (make(c), c) for n, (_, c) in pairs.items()}
        methods = blur_methods("Motion" if "motion" in cond else "Gaussian") if "blur" in cond else dict(METHODS)
        rows, outs = run(methods, scenes)
        all_rows += [dict(r, condition=cond) for r in rows]
        section(f"2. Simulated: {cond}", rows, md)
        if "motion" in cond:
            s = next(iter(scenes))
            montage([scenes[s][0], outs[(s, FULL)], outs[(s, AUTO)], scenes[s][1]],
                    ["fog + motion blur", "full pipeline", "+ Wiener (blur estimated)", "clear photo"],
                    OUT / "figures" / "simulated_motion_blur_montage.png")
    write_csv(OUT / "simulated.csv", all_rows)

    # 3) urban: synthetic city streets, fog from the true depth map
    md += ["## 3. Urban test set: synthetic city streets with true depth", "",
           f"{len(urban)} synthetic streets (assets/test images/urban_synthetic, 640x400, different building layouts, "
           "cars, lane markings), fogged with t = exp(-beta * depth) from their true depth map.", ""]
    urban_rows, trans = [], transmission_rows(urban)
    for cond, beta, blur in (("fog", 1.6, None), ("fog + horizontal motion blur (2 px)", 1.6, "Motion"),
                             ("fog + Gaussian blur (2 px)", 1.6, "Gaussian")):
        scenes = {}
        for n, (c, d) in urban.items():
            f = C.simulate_fog(c, d, beta)
            scenes[n] = (C.simulate_blur(f, 2.0, blur) if blur else f, c)
        rows, outs = run(blur_methods(blur) if blur else dict(METHODS), scenes)
        urban_rows += [dict(r, condition=cond) for r in rows]
        section(f"3.{len(set(r['condition'] for r in urban_rows))}. Urban: {cond} (beta {beta})", rows, md, "###")
        if cond == "fog":
            s = next(iter(scenes))
            montage([scenes[s][0], outs[(s, "Dark channel prior only")], outs[(s, FULL)], scenes[s][1]],
                    ["foggy city", "dark channel only", "full pipeline", "clear city"], OUT / "figures" / "urban_montage.png")
            montage([lines_overlay(scenes[s][0]), lines_overlay(outs[(s, FULL)]), lines_overlay(scenes[s][1])],
                    ["Hough lines: foggy", "Hough lines: full pipeline", "Hough lines: clear"],
                    OUT / "figures" / "downstream_lines_urban.png")
    write_csv(OUT / "urban.csv", urban_rows)

    # fog density sweep (true depth)
    md += ["### Fog density sweep (urban set, full pipeline vs doing nothing)", "",
           "| Beta | Input PSNR | Full PSNR | Input SSIM | Full SSIM | Input edge F1 | Full edge F1 |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for beta in (0.8, 1.6, 2.4, 3.2):
        scenes = {n: (C.simulate_fog(c, d, beta), c) for n, (c, d) in urban.items()}
        rows, _ = run({INPUT: OFF, FULL: C.P_BALANCED}, scenes)
        g = lambda m, k: np.mean([r[k] for r in rows if r["method"] == m])
        md.append(f"| {beta:g} | {g(INPUT, 'psnr'):.1f} | {g(FULL, 'psnr'):.1f} | {g(INPUT, 'ssim'):.3f} | {g(FULL, 'ssim'):.3f} "
                  f"| {g(INPUT, 'edge_f1'):.3f} | {g(FULL, 'edge_f1'):.3f} |")
    md += ["", "### Transmission estimate against the true depth", "",
           "Mean over the urban set; the true transmission is exp(-beta * depth), clipped to the same minimum as the estimate.", "",
           "| Beta | Transmission MAE | Correlation with true t | Airlight error (0-255) |", "| ---: | ---: | ---: | ---: |"]
    for beta in (0.8, 1.6, 2.4, 3.2):
        sel = [r for r in trans if r["beta"] == beta]
        md.append(f"| {beta:g} | {np.mean([r['t_mae'] for r in sel]):.3f} | {np.mean([r['t_corr'] for r in sel]):.3f} "
                  f"| {np.mean([r['airlight_err'] for r in sel]):.1f} |")
    md.append("")
    write_csv(OUT / "transmission.csv", trans)

    # 5) blur estimation
    est_rows, est_md = blur_estimation({n: c for n, (_, c) in pairs.items()}, {n: c for n, (c, _) in urban.items()})
    write_csv(OUT / "blur_estimation.csv", est_rows)
    hazy_est = [C.estimate_blur(h) for h, _ in pairs.values()]
    md += ["## 4. Blur estimation", "",
           "`estimate_blur` measures the width of the strongest edges across and along the image and subtracts the width of a sharp "
           "photo. Each row blurs the clear photos by a known amount (fog added first, beta 1.6 unless the row says otherwise) and "
           "compares the estimate with it. The Gaussian calibration factor was fit on the real photos, so the synthetic cities are an "
           "independent check. Motion blur means horizontal motion, length 3 x amount.", "", est_md, "",
           f"On the {len(pairs)} real hazy photos (true blur unknown) the estimator found: "
           f"{', '.join(f'{k} {sum(e['kind'] == k for e in hazy_est)}' for k in dict.fromkeys(e['kind'] for e in hazy_est))} "
           f"(largest estimate {max(e['amount'] for e in hazy_est):.2f} px).", ""]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
