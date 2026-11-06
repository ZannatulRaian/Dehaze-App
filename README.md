<div align="center">

# Dehaze App

**Upload a foggy city photo, tune an enhancement pipeline built on frequency-domain enhancement, histogram stretching and adaptive smoothing and sharpening, check the result with measurements, and download it.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-dehazeapp.streamlit.app-C8102E?style=for-the-badge&logo=streamlit&logoColor=white)](https://dehazeapp.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-DIP-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

</div>

---

## 📷 Main View

![Dehaze App preview](assets/Preview/preview.gif)

---

## Overview

**Dehaze App** is a Streamlit web app that makes buildings, roads and vehicles more visible in foggy or blurred urban photos. It was built as a Digital Image Processing course project (*Project 7*).

The project asks for three modules, and all three are implemented as separate stages with their own controls:

| Project requirement | Where it is in the app |
| --- | --- |
| Frequency-domain enhancement (edge sharpening) | Stage 04: FFT high-frequency emphasis with a Gaussian high-pass |
| Histogram stretching | Stage 03: percentile stretch of luminance |
| Adaptive smoothing and sharpening | Stages 05 and 06: edge-preserving smoothing, then capped unsharp masking |
| Simulate foggy images | Input stage: fog from the atmospheric scattering model, plus optional Gaussian or motion blur |
| Or use real ones | Upload any photo (tick *Photo is already foggy*); the bundled outdoor test photos come with clear references that `evaluate.py` uses |
| Make urban structures more visible | **Edges** view, edge-pixel and gradient measurements, before/after histograms |
| Blurred or foggy view | Fog removal (dark channel prior) and Wiener deblurring (stage 02) |
| Traffic monitoring, construction inspection | **Batch Processing** card: many photos at once, time per photo, ZIP download |

The **Course modules only** switch runs just the three project modules, so their effect can be shown on its own. The **Dark Channel Prior** and the optional **Wiener deblur** are additions that handle fog and blur, which the three modules alone cannot remove.

## Live demo

👉 **https://dehazeapp.streamlit.app/** (this may show an older version until it is redeployed from this code)

## What is in the app

Everything is on the main page (no sidebar), under the pixel-art header strip.

**Preview card** (left)
- Upload a photo by dropping it on the box. **Change Image** starts over.
- **View** switch:
  - **Side by side** — original and dehazed.
  - **Slider** — one image with a divider you drag between original (left) and dehazed (right).
  - **Original** and **Dehazed** — one image at a time.
  - **Stages** — steps through the pipeline in the order it runs. *Cumulative* shows the image before and after the stage you pick; *This stage only* shows the original next to the original with just that stage applied.
  - **Edges** — the Sobel gradient of the original and the dehazed image on one shared scale, so you can see how much more of the buildings and roads shows up.
- **Measurements**: contrast, entropy, mean gradient, edge pixels, processing time, and PSNR and SSIM when fog or blur is simulated on a clear photo (that photo is the reference), plus brightness and R, G, B histogram comparisons.
- The card stays in view while you scroll the pipeline controls.

**Enhancement Pipeline card** (right)
- **00 Input** — *Photo is already foggy*, simulated fog density, simulated blur (Gaussian or horizontal motion), and the **Course modules only** switch.
- **01 Dark Channel Prior** — on/off, haze removed (ω), minimum transmission (t₀), patch size.
- **02 Deblur (Wiener)** — blur to undo (0 = off), assumed blur type, noise damping.
- **03 Histogram Stretch**, **04 Frequency-Domain Filter** (strength and cutoff radius), **05 Adaptive Smoothing**, **06 Adaptive Sharpening**.
- **07 Final Tone** — gamma, saturation and a gray-world colour balance.
- **Preset** (Balanced, Light haze, Dense fog, Maximum detail, Fog removal only, Course modules only), **Auto-Enhance**, **Reset**, **Download Result**.

**Batch Processing** (below): upload several photos, process them with the current settings, see time and contrast per photo, and download everything as a ZIP with a CSV of the results.

**Method** (bottom): the fog model, the approach and the references.

## How it works

The pipeline (in `dehaze_core.py`) runs in this order; the numbers in brackets are the stage numbers shown in the app:

| Step | Stage | What it does |
| ---- | ----- | ------------ |
| 1 | **Dark Channel Prior** [01] (on by default) | Estimates the airlight and a per-pixel transmission map, refined with a **guided filter** so it follows real edges, then recovers the haze-free scene `J = (I − A) / t + A` |
| 2 | **Deblur (Wiener)** [02] (off at 0) | Wiener deconvolution of luminance in the Fourier domain for an assumed Gaussian or horizontal motion blur; mirror padding and a noise-damping term limit ringing |
| 3 | **Adaptive Smoothing** [05] | Edge-preserving bilateral filter; smooths flat areas and the noise fog removal amplifies, keeps edges |
| 4 | **Adaptive Sharpening** [06] | Unsharp mask on luminance with the added detail capped, so strong edges don't overshoot into halos |
| 5 | **Histogram Stretch** [03] | Linear contrast stretch of luminance, clipping 0–2 % from each end |
| 6 | **Frequency-Domain Filter** [04] | FFT high-frequency emphasis with a smooth Gaussian high-pass (no ringing) and mirror padding (no border wrap-around) |
| 7 | **Final Tone** [07] | Optional gray-world colour balance, saturation and gamma |

**Design notes**

- Stages 2–6 work on **luminance only** (the Y channel of YCrCb). Processing R, G and B separately shifts hues and paints coloured noise.
- Every stage returns its input untouched when it is switched off or its strength is `0`, so **Reset** switches the four filters and the deblur off and puts the tone controls back to neutral. **Course modules only** switches off the dark channel, deblur and tone stages without touching their settings.
- Images larger than **1200 px** on the long edge are downscaled before processing.
- The image-processing code is in `dehaze_core.py` (no Streamlit), so it can be tested and evaluated on its own.

## Default settings and evaluation

The first version of the defaults over-processed: on the seven real foggy test scenes it scored **worse than doing nothing** on structural similarity (SSIM 0.58 against 0.69 for the untouched photo). The defaults were then chosen by a random parameter search on four of the scenes, with the three course modules required to stay active, the tone stage kept neutral apart from a mild colour balance, and a minimum contrast and edge gain required. They kept their advantage on the three scenes left out of the search. The same search with only the three modules allowed gives the **Course modules only** preset. The other presets are hand-set around the default.

`python evaluate.py` re-runs the experiment and writes `results/` (CSV files, `summary.md`, comparison images). Each method is an ablation, measured against the clear photo of each scene; values are means over the 7 scenes at up to 800 px.

**Real fog** (7 hazy photos with clear references)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 13.7 | 0.674 | 1.00 | 13.0 | 0.00 |
| Histogram stretching only | 12.1 | 0.701 | 2.43 | 35.4 | 0.01 |
| Frequency-domain enhancement only | 13.8 | 0.739 | 1.08 | 25.7 | 0.05 |
| Adaptive smoothing + sharpening only | 13.7 | 0.602 | 1.01 | 8.3 | 0.04 |
| The three course modules together | 13.7 | 0.684 | 2.07 | 30.8 | 0.08 |
| Dark channel prior only | 17.0 | 0.769 | 1.17 | 22.7 | 0.09 |
| Full pipeline (Balanced default) | 15.9 | 0.706 | 2.03 | 34.6 | 0.10 |

**Simulated fog** (the 7 clear photos, fogged with β = 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.2 | 0.657 | 1.00 | 15.5 | 0.00 |
| Histogram stretching only | 10.5 | 0.765 | 2.25 | 37.4 | 0.01 |
| Frequency-domain enhancement only | 10.2 | 0.732 | 1.07 | 29.4 | 0.05 |
| Adaptive smoothing + sharpening only | 10.2 | 0.569 | 1.01 | 9.9 | 0.05 |
| The three course modules together | 11.0 | 0.694 | 1.99 | 34.2 | 0.09 |
| Dark channel prior only | 21.5 | 0.943 | 1.25 | 36.0 | 0.10 |
| Full pipeline (Balanced default) | 18.0 | 0.805 | 1.91 | 47.0 | 0.11 |

**Simulated fog + horizontal motion blur** (2 px)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.1 | 0.478 | 1.00 | 3.0 | 0.00 |
| Histogram stretching only | 10.3 | 0.564 | 2.37 | 17.0 | 0.01 |
| Frequency-domain enhancement only | 10.1 | 0.519 | 1.03 | 9.1 | 0.04 |
| Adaptive smoothing + sharpening only | 10.1 | 0.444 | 1.00 | 1.9 | 0.04 |
| The three course modules together | 10.7 | 0.555 | 2.14 | 16.8 | 0.08 |
| Dark channel prior only | 19.9 | 0.681 | 1.24 | 13.3 | 0.09 |
| Full pipeline (Balanced default) | 17.2 | 0.655 | 1.92 | 25.7 | 0.10 |
| Full pipeline + Wiener deblur (blur known) | 17.9 | 0.782 | 1.93 | 44.4 | 0.14 |

What the numbers show:

- The **three course modules** clearly make structure more visible: contrast roughly doubles and edge pixels go from about 13 % to 31 % on real fog. Used alone they barely improve PSNR (13.7 dB, the same as the foggy photo) and SSIM only slightly (0.684 against 0.674), because they cannot remove the haze itself.
- **Histogram stretching** and **frequency-domain enhancement** each improve SSIM on their own. **Adaptive smoothing + sharpening** on its own does not (SSIM 0.602 against 0.674, and fewer edge pixels), so it is best used together with the other stages.
- The **dark channel prior** does most of the work for fidelity: PSNR 13.7 → 17.0 dB on real fog and 10.2 → 21.5 dB on simulated fog. Adding the three modules to it lowers PSNR and SSIM a little but raises contrast and edge pixels, which is the trade-off the *Balanced* default and the **Fog removal only** preset let you choose between.
- **Wiener deblurring** helps when the blur is known: with motion blur, SSIM rises from 0.655 to 0.782 and PSNR from 17.2 to 17.9 dB.
- A full pipeline run takes about **0.1 s per 800 px photo** on the CPU of the machine used for the evaluation, with no GPU.

**Limitations**

- Only seven real scenes with clear references are available, and the defaults were tuned on a subset of them, so the numbers are indicative, not a benchmark.
- PSNR and SSIM measure similarity to the clear photo, not how visible structures look to a person; contrast and edge measures rise with over-sharpening too. Judge the pictures as well.
- Deblurring assumes the blur type and size are known; there is no blind blur estimation.
- Fog simulated on a photo uses a rough depth guess (farther toward the top of the frame).
- Single images and batches only: no video, and no detection of buildings, roads or vehicles.

## Run locally

```bash
git clone https://github.com/ZannatulRaian/Dehaze-App.git
cd Dehaze-App

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run dehaze_app.py
```

The app opens at `http://localhost:8501`.

```bash
python evaluate.py    # re-run the evaluation, writes ./results
python test_core.py   # quick checks of the image-processing code
```

**Optional:** to open iPhone HEIC/HEIF photos, run `pip install pillow-heif`.

## Project structure

```
Dehaze-App/
├── dehaze_app.py          # Streamlit app (layout, controls, views, batch)
├── dehaze_core.py         # Image processing: stages, fog/blur simulation, metrics (no Streamlit)
├── evaluate.py            # Ablation study on the test scenes -> results/
├── test_core.py           # Checks for the core
├── results/               # Evaluation output: CSV files, summary.md, comparison images
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit theme/config
└── assets/
    ├── cars/              # Pixel-art car sprites (header strip)
    ├── clouds/            # Pixel-art cloud sprites (header strip)
    ├── test images/       # Sample hazy photos to upload and try
    └── Preview/           # Demo recording (preview.gif + full .mp4 recording)
```

## Try it with the sample images

`assets/test images/` includes ready-to-use hazy photos. Drop any of them on the upload box:

- `city_smog/` – smoggy city streets and a foggy park
- `outdoor_fog_with_clear_reference/` – hazy scenes, each paired with a clear reference photo; `evaluate.py` measures them against their references

## Tech stack

[Streamlit](https://streamlit.io/) · [OpenCV](https://opencv.org/) (headless) · [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) · [Pillow](https://python-pillow.org/)

## References

- K. He, J. Sun, X. Tang, *Single Image Haze Removal Using Dark Channel Prior*, CVPR 2009 (IEEE TPAMI 2011)
- K. He, J. Sun, X. Tang, *Guided Image Filtering*, ECCV 2010 (IEEE TPAMI 2013)
- S. G. Narasimhan, S. K. Nayar, *Vision and the Atmosphere*, IJCV 2002
- H. Koschmieder, *Theorie der horizontalen Sichtweite*, 1924
- R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018
- C. Tomasi, R. Manduchi, *Bilateral Filtering for Gray and Color Images*, ICCV 1998
- Z. Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE TIP 2004

## Author

**Zannatul Raian** · [@ZannatulRaian](https://github.com/ZannatulRaian)
