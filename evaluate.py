"""Evaluation of the Dehaze App on the bundled test scenes.

    python evaluate.py            # writes everything to ./results

Two experiments, both measured against the clear reference photo of each scene:

  1. REAL fog   - the 7 hazy photos in assets/test images/outdoor_fog_with_clear_reference.
  2. SIMULATED  - the 7 clear photos, fogged with the scattering model I = J*t + A*(1-t)
                  (beta = 1.6), optionally also blurred (Gaussian or horizontal motion).

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
DATA = ROOT / "assets" / "test images" / "outdoor_fog_with_clear_reference"
OUT = ROOT / "results"

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


def load_pairs(max_dim):
    pairs = {}
    for hazy in sorted(DATA.glob("scene*_hazy.jpg")):
        name = hazy.stem.replace("_hazy", "")
        h = cv2.cvtColor(cv2.imread(str(hazy)), cv2.COLOR_BGR2RGB)
        c = cv2.cvtColor(cv2.imread(str(DATA / f"{name}_clear_reference.jpg")), cv2.COLOR_BGR2RGB)
        s = min(1.0, max_dim / max(h.shape[:2]))
        h = cv2.resize(h, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        c = cv2.resize(c, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_AREA)
        pairs[name] = (h, c)
    return pairs


def run(methods, scenes):
    """scenes: {name: (input, clear)} -> list of result rows"""
    rows, outputs = [], {}
    for scene, (inp, clear) in scenes.items():
        base = C.quality_metrics(inp)
        for method, P in methods.items():
            t = time.perf_counter()
            out = C.dehaze_pipeline(inp, P)
            sec = time.perf_counter() - t
            q = C.quality_metrics(out)
            rows.append(dict(scene=scene, method=method, psnr=C.psnr(out, clear), ssim=C.ssim(out, clear),
                             contrast_x=q["contrast"] / base["contrast"], edges_pct=q["edges"], seconds=sec))
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


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def montage(images, labels, path, width=360):
    tiles = []
    for img, lab in zip(images, labels):
        t = cv2.resize(img, (width, int(width * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        bar = np.full((28, width, 3), 255, np.uint8)
        cv2.putText(bar, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, t]))
    h = min(t.shape[0] for t in tiles)
    cv2.imwrite(str(path), cv2.cvtColor(np.hstack([t[:h] for t in tiles]), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dim", type=int, default=800, help="downscale test photos to this long edge")
    args = ap.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.max_dim)
    md = ["# Evaluation results", "",
          f"{len(pairs)} scenes, photos scaled to at most {args.max_dim}px on the long edge. "
          "PSNR and SSIM are measured against the clear reference photo of each scene; the contrast gain is "
          "relative to the foggy input and edge pixels are the share of pixels with a gradient above 12. "
          "Each value is the mean over the scenes.", ""]

    # 1) real fog
    rows, outs = run(METHODS, pairs)
    write_csv(OUT / "real_fog.csv", rows)
    md += ["## 1. Real fog", "", md_table(summarise(rows)), ""]
    s = "scene12" if "scene12" in pairs else next(iter(pairs))
    montage([pairs[s][0], outs[(s, "The three course modules together")], outs[(s, "Full pipeline (Balanced default)")], pairs[s][1]],
            ["foggy photo", "three modules only", "full pipeline", "clear reference"], OUT / "figures" / "real_fog_montage.png")

    # 2) simulated fog and blur on the clear photos
    conditions = {
        "fog": lambda c: C.simulate_fog(c, None, 1.6),
        "fog + Gaussian blur (2 px)": lambda c: C.simulate_blur(C.simulate_fog(c, None, 1.6), 2.0, "Gaussian"),
        "fog + horizontal motion blur (2 px)": lambda c: C.simulate_blur(C.simulate_fog(c, None, 1.6), 2.0, "Motion"),
    }
    all_rows = []
    for cond, make in conditions.items():
        scenes = {n: (make(c), c) for n, (_, c) in pairs.items()}
        methods = dict(METHODS)
        if "blur" in cond:
            kind = "Motion" if "motion" in cond else "Gaussian"
            methods["Full pipeline + Wiener deblur (blur known)"] = dict(C.P_BALANCED, deblur=2.0, deblur_kind=kind)
        rows, outs = run(methods, scenes)
        all_rows += [dict(r, condition=cond) for r in rows]
        md += [f"## 2. Simulated: {cond}", "", md_table(summarise(rows)), ""]
        if "motion" in cond:
            s = next(iter(scenes))
            montage([scenes[s][0], outs[(s, "Full pipeline (Balanced default)")],
                     outs[(s, "Full pipeline + Wiener deblur (blur known)")], scenes[s][1]],
                    ["fog + motion blur", "full pipeline", "+ Wiener deblur", "clear photo"],
                    OUT / "figures" / "simulated_motion_blur_montage.png")
    write_csv(OUT / "simulated.csv", all_rows)
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
