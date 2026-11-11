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

## Project Brief

**Project 7. Urban Fog Cleaner (Dehazing Tool)**

**Idea:** Simulate foggy images (or use real ones) and use **image enhancement techniques** to make urban structures more visible (e.g., buildings, roads) from a blurred or foggy view.

**Modules Used:**

- Frequency Domain Enhancement (for edge sharpening)
- Histogram Stretching
- Adaptive Smoothing and Sharpening

**Impact:** Useful for traffic monitoring systems or construction inspection under poor visibility.

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
| Or use real ones | Upload any photo (tick *Photo is already foggy*); the bundled outdoor test photos come with clear references that `evaluate.py` uses, and `assets/test images/urban_synthetic/` adds synthetic city streets with clear versions and true depth maps |
| Make urban structures more visible | **Edges** view, edge-pixel and gradient measurements, Canny edge and Hough line counts, before/after histograms |
| Blurred or foggy view | Fog removal (dark channel prior) and Wiener deblurring with automatic blur-size estimation (stage 02) |
| Traffic monitoring, construction inspection | **Traffic monitoring** preset, **Batch Processing** card (many photos, ZIP download) and **Video** card (temporal smoothing) |

The **Course modules only** switch runs just the three project modules, so their effect can be shown on its own. The **Dark Channel Prior** and the optional **Wiener deblur** are additions that handle fog and blur, which the three modules alone cannot remove. **CLAHE** (local contrast) and **colour restoration** are further additions that work on what is left after the fog is gone.

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
  - **Transmission** — the original next to the dark channel's estimated transmission map as a colour map (dark = thick haze or far away, bright = clear or near).
  - **Modules** — the three course modules made visible: the FFT spectrum of the original and the dehazed image on one scale, their mean spectrum by frequency next to the frequency filter's gain curve, the luminance histogram before the stretch with its clip points marked and after it, and two maps showing where the adaptive smoothing changes pixels and how strongly the adaptive sharpening acts.
  - **Compare** — the full pipeline next to single-technique baselines (histogram equalisation, CLAHE only, gamma only, plain unsharp mask, dark channel prior only), each with its contrast and edge share, and PSNR and SSIM when a clear reference exists.
  - **Applications** — *Traffic monitoring*: vehicles and people found by a YOLOv5n detector before and after, drawn as boxes (red cars and trucks, orange motorcycles and people), with a confidence slider and a *tiles* switch that also searches overlapping zoomed-in tiles for small, distant vehicles and motorcycles (needs `onnxruntime`, see `requirements-eval.txt`; without it the tab says so). *Construction inspection*: Canny edges (red) and Hough lines (green) before and after, with the counts.
- **Measurements**: contrast, entropy, mean gradient, edge pixels, processing time, how many edges a standard Canny detector finds and how many building (vertical) and lane (oblique) lines a Hough line detector finds, and, when fog or blur is simulated on a clear photo (that photo is the reference), PSNR, SSIM, edge F1 and line recall against it, plus brightness and R, G, B histogram comparisons. Three no-reference rows compare the result with the foggy input: visible-edge gain e, gradient ratio and newly clipped pixels.
- Under the measurements: **Download before/after PNG** and **Download measurements CSV**.
- The card stays in view while you scroll the pipeline controls. An image enlarged with its hover button covers the whole window.

**Enhancement Pipeline card** (right)
- **00 Input** — *Photo is already foggy*, simulated fog density, simulated blur (Gaussian or motion at any angle), and the **Course modules only** switch.
- **01 Dark Channel Prior** — on/off, haze removed (ω), minimum transmission (t₀), patch size, and an **adaptive airlight and t₀** switch (airlight from the brightest fifth of the haze candidates instead of one pixel; t₀ scaled to the measured haze).
- **02 Deblur (Wiener)** — blur to undo (0 = off), blur type, noise damping, or **Detect the blur automatically**, which measures the blur type and size from the photo and shows what it found.
- **03 Histogram Stretch**, **03b CLAHE** (clip strength and tile grid), **04 Frequency-Domain Filter** (strength and cutoff radius), **05 Adaptive Smoothing**, **06 Adaptive Sharpening** (noise-aware).
- **07 Final Tone** — gamma, saturation, a gray-world colour balance and **colour restoration** (mid-tone white balance plus a saturation boost that is larger the more washed-out the image still is).
- **Preset** (Balanced, Light haze, Dense fog, Maximum detail, Fog removal only, Course modules only, Traffic monitoring), **Auto-Enhance**, **Reset**, **Download Result**.

**Video** card: upload a short clip, choose how many frames to process and whether to smooth over time (less flicker), then watch it as **Side by side** (original left, dehazed right, in sync), **Dehazed** or **Original**. **Applications** runs the traffic detection and the construction edge and line overlay on any of 12 sample frames (with an optional vehicle count over all of them), **Live detection** runs the vehicle detector on every frame of the whole clip (with a live preview while it runs) and plays the result as a video with boxes and counts, original left and dehazed right, with a per-frame count chart and a download, and **Graphs** plots contrast, edge pixels and brightness per frame plus the histogram, gamma and spectrum graphs of the picked frame. Download the dehazed mp4 below.

**Batch Processing** (below): upload several photos, process them with the current settings, then switch between **Table** (time and contrast per photo), **All images** (every original next to its dehazed version), **Side by side** and **Slider** (pick one photo), **Applications** (traffic detection and construction overlay for the picked photo, with an optional vehicle count for every photo) and **Graphs** (contrast and edge gain per photo, plus histogram, gamma and spectrum graphs of the picked photo), and download everything as a ZIP with a CSV of the results.

**Method** (bottom): the fog model, the approach and the references.

## How it works

The pipeline (in `dehaze_core.py`) runs in this order; the numbers in brackets are the stage numbers shown in the app:

| Step | Stage | What it does |
| ---- | ----- | ------------ |
| 1 | **Dark Channel Prior** [01] (on by default) | Estimates the airlight and a per-pixel transmission map, refined with a **guided filter** so it follows real edges, then recovers the haze-free scene `J = (I − A) / t + A`. With *adaptive* on, the airlight is the mean of the brightest fifth of the haze candidates and the floor t₀ is scaled to the measured haze (`t₀ · clip(1.5 − 1.6·haze, 0.6, 1.4)`) |
| 2 | **Deblur (Wiener)** [02] (off at 0) | Wiener deconvolution of luminance in the Fourier domain for a Gaussian or linear motion blur (any angle) whose size you set or that `estimate_blur` measures; mirror padding and a noise-damping term limit ringing |
| 3 | **Adaptive Smoothing** [05] | Edge-preserving bilateral filter; smooths flat areas and the noise fog removal amplifies, keeps edges |
| 4 | **Adaptive Sharpening** [06] | Unsharp mask on luminance with the added detail capped, so strong edges don't overshoot into halos. The detail is weighted by a local-activity map and an Immerkaer noise estimate, so edges and textures are sharpened fully and flat, noisy areas much less |
| 5 | **Histogram Stretch** [03] | Linear contrast stretch of luminance, clipping 0–2 % from each end |
| 6 | **CLAHE** [03b] | Contrast-limited adaptive histogram equalisation on the L channel of CIELAB (clip limit 1–4, tile grid 2–16), blended with its input by the strength; recovers local contrast that the global stretch cannot |
| 7 | **Frequency-Domain Filter** [04] | FFT high-frequency emphasis with a smooth Gaussian high-pass (no ringing) and mirror padding (no border wrap-around) |
| 8 | **Final Tone** [07] | Colour restoration (mid-tone white balance, adaptive saturation), optional gray-world colour balance, saturation and gamma |

**Design notes**

- Stages 2–7 work on **luminance only** (the Y channel of YCrCb; CLAHE uses the L channel of CIELAB). Processing R, G and B separately shifts hues and paints coloured noise.
- Every stage returns its input untouched when it is switched off or its strength is `0`, so **Reset** switches the filters, CLAHE and the deblur off and puts the tone controls back to neutral. **Course modules only** switches off the dark channel, deblur and tone stages without touching their settings.
- **Blur estimation** (`estimate_blur`): the width of the strongest edges is measured along 8 directions (the std of the edge's gradient profile, which equals the blur kernel's std). A sharp photo's own edge width is subtracted in quadrature; equal excess in all directions is Gaussian blur, excess that peaks in one direction is motion blur at that angle (box length = 3 x amount). Fog scales an edge's step and its gradient by the same factor, so the estimate does not depend on fog density.
- Images larger than **1200 px** on the long edge are downscaled before processing.
- The image-processing code is in `dehaze_core.py` (no Streamlit), so it can be tested and evaluated on its own.

## Default settings and evaluation

Charts (SVG, regenerate with `python evaluate.py --charts`, about 15 s): [methods against simple baselines (SSIM, edge F1)](results/figures/chart_methods_ssim_f1.svg), [PSNR](results/figures/chart_methods_psnr.svg), [ablation](results/figures/chart_ablation.svg), fog-density sweeps ([garden photos](results/figures/chart_sweep_7_garden_photos.svg), [synthetic streets](results/figures/chart_sweep_synthetic_streets.svg), [real streets](results/figures/chart_sweep_9_real_streets.svg)) and [runtime against image size](results/figures/chart_runtime.svg). The sweep charts read `results/strength_sweep.csv`, which a full `python evaluate.py` run writes.

The first version of the defaults over-processed: on the seven real foggy test scenes it scored **worse than doing nothing** on structural similarity (SSIM 0.58 against 0.69 for the untouched photo). The defaults were then chosen by a random parameter search on four of the scenes, with the three course modules required to stay active. The same search with only the three modules allowed gives the **Course modules only** preset. The other presets are hand-set around the default.

Four additions were made later (CLAHE, the adaptive dark channel, noise-aware sharpening, colour restoration) and the whole evaluation was re-run with them on. They were **not** part of the parameter search; CLAHE (strength 15) and colour restoration (30) were set by hand. `evaluate.py` also measures the **previous defaults** (`P_PREVIOUS` in `dehaze_core.py`: the same pipeline with the four additions off), so each table below can be read against them.

`python evaluate.py` re-runs everything and writes `results/` (CSV files, `summary.md`, comparison images; about 10 minutes with `--fit-strength`, about 7 without). Every method is measured against the clear photo of each scene. There are four kinds of test data:

| Test data | What it is | What it can and cannot show |
| --- | --- | --- |
| **Real fog** | 7 hazy garden and park photos, each with a clear reference | Real fog, but not city streets |
| **Real streets** | 10 clear photos of real streets (`python get_street_photos.py` downloads 9: 8 from the Mask R-CNN sample images and a bus scene from the ultralytics sample images; one more is bundled), fogged with the scattering model and a rough depth guess | Real street content (cars, people, buildings), but simulated fog |
| **Synthetic streets** | 6 renders in `assets/test images/urban_synthetic/` (buildings, cars, lane lines) with a clear version and the **true depth map** | Exact ground truth; clean flat colours flatter edge detection |
| **Synthetic traffic clip** | 30 frames, fixed camera, 3 cars, fog from the true depth, sensor noise | Video behaviour |

**No real hazy/clear photo pairs of city streets are bundled**, because none were available. `assets/test images/urban_real_foggy/` has 7 real foggy or smoggy city photos, but without clear references, so they can be judged by eye and by the no-reference rows (visible-edge gain e, gradient ratio), not by PSNR or SSIM. To use real pairs, put `name_hazy.jpg` and `name_clear_reference.jpg` in `assets/test images/urban_with_clear_reference/`; `evaluate.py` adds a section for them.

### Visibility and fidelity (SSIM and PSNR against the clear photo)

| Method | Real fog: SSIM | PSNR | Real streets, fog 1.6: SSIM | PSNR | Synthetic streets: SSIM | PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input | 0.674 | 13.7 | 0.647 | 10.9 | 0.754 | 12.2 |
| The three course modules together | 0.685 | 13.7 | 0.755 | 14.3 | 0.696 | 12.2 |
| Dark channel prior only | 0.774 | 16.9 | 0.912 | 20.4 | 0.871 | 21.8 |
| Previous defaults (four additions off) | 0.706 | 15.9 | 0.862 | 18.7 | 0.793 | 19.4 |
| Full pipeline (Balanced default) | 0.692 | 16.0 | 0.845 | 19.0 | 0.806 | 19.3 |
| Full pipeline, strength set from the haze level | 0.717 | 15.6 | 0.847 | 18.6 | 0.837 | 17.3 |
| Traffic monitoring preset | 0.735 | 14.9 | 0.920 | 20.5 | 0.866 | 20.2 |

Visibility without a clear photo (**visible-edge gain e** and **gradient ratio r**, Hautière et al., against the foggy input; Balanced default against the previous defaults): real fog e +1.43 against +1.11 and r 3.73 against 3.21; real streets e +0.56 against +0.47 and r 3.62 against 3.24; synthetic streets e +1.30 against +1.16. The four additions raise visibility and lower fidelity a little (SSIM −0.014 on real fog, −0.017 on real streets, +0.013 on synthetic streets; PSNR within ±0.3 dB). The ablation below shows that almost all of that comes from CLAHE.

### Impact test: what standard detectors find before and after

Canny (fixed 50/150) and a Hough line detector are run on each result and compared with what they find in the clear photo. **Edge F1** (2 px tolerance) and **line recall** are the numbers to read; raw counts rise with texture and noise too. On the real streets a **YOLOv5n** detector (`detect.py`, evaluation only, weights downloaded on first use, AGPL licence) also counts vehicles and people; *detection recall* is the share of what it finds in the clear photo that it still finds (IoU 0.5).

| Method | Real fog: edge F1 | line recall | Real streets: edge F1 | line recall | vehicles/people found (clear photo: 11.1) | detection recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input | 0.155 | 0.039 | 0.450 | 0.085 | 9.3 | 0.743 |
| Histogram stretching only | 0.642 | 0.443 | 0.818 | 0.404 | 10.4 | 0.848 |
| The three course modules together | 0.661 | 0.520 | 0.846 | 0.527 | 8.3 | 0.703 |
| Dark channel prior only | 0.462 | 0.161 | 0.880 | 0.462 | 10.5 | 0.890 |
| Previous defaults (four additions off) | 0.645 | 0.551 | 0.878 | 0.640 | 9.3 | 0.752 |
| Full pipeline (Balanced default) | 0.612 | 0.573 | 0.847 | 0.643 | 9.2 | 0.754 |
| Full pipeline, strength set from the haze level | 0.662 | 0.526 | 0.871 | 0.649 | 9.4 | 0.803 |
| **Traffic monitoring preset** | 0.664 | 0.399 | **0.925** | 0.574 | **10.9** | **0.906** |

Findings:

- On real fog the standard detectors find almost nothing in the photo (0.4 % edge pixels, 6 lines) and a lot after processing; edge F1 improves in every scene for every method (7 of 7).
- **The full pipeline's strong sharpening and frequency stages find more edges and lines but *lose* vehicles**: detection recall 0.754 against 0.743 for the untouched photo and 0.890 for the dark channel alone, because over-sharpened and over-stretched cars no longer look like cars to the detector. For traffic monitoring this is the result that matters, so there is a **Traffic monitoring** preset (fog removal + histogram stretching only, strength set from the haze level in the app). It gives the best edge F1 and detection recall on the real streets (0.925 and 0.906; the strength-adaptive variant is 0.908 and 0.900).
- The preset's gain was checked on two disjoint halves of the street photos at three fog densities and held in each, for the earlier defaults; that check was not repeated for this run.
- Building (near-vertical) and lane (oblique) line counts are split out in `summary.md`. The three course modules raise lane-line counts from 8 to 96 on the real streets.
- With only 10 street photos, differences of a few hundredths in detection recall are within noise.

### Do the three course modules matter next to the dark channel prior?

Each row removes one part of the Balanced pipeline (`summary.md`, section 6):

| Pipeline | Real fog SSIM / F1 | Real streets SSIM / F1 | Detection recall |
| --- | ---: | ---: | ---: |
| Full pipeline | 0.692 / 0.612 | 0.845 / 0.847 | 0.754 |
| without histogram stretching | 0.766 / 0.670 | 0.849 / 0.861 | 0.732 |
| without frequency-domain enhancement | 0.727 / 0.644 | 0.884 / 0.889 | 0.770 |
| without adaptive smoothing + sharpening | 0.685 / 0.616 | 0.890 / 0.883 | 0.803 |
| without all three modules | 0.808 / 0.664 | 0.919 / 0.918 | 0.858 |
| without the three modules **and CLAHE** | 0.790 / **0.519** | 0.912 / 0.881 | 0.850 |
| without the dark channel prior | 0.688 / 0.664 | **0.778** / 0.859 | 0.756 |
| without CLAHE | 0.701 / 0.638 | 0.860 / 0.878 | 0.774 |
| without colour restoration | 0.693 / 0.615 | 0.845 / 0.848 | 0.740 |
| without the adaptive dark channel | 0.695 / 0.615 | 0.846 / 0.851 | 0.754 |
| without noise-aware sharpening | 0.692 / 0.613 | 0.846 / 0.848 | 0.754 |
| Previous defaults (none of the four additions) | 0.706 / 0.645 | 0.862 / 0.878 | 0.752 |

The honest reading: the **dark channel prior carries fidelity** (remove it and street SSIM falls 0.845 to 0.778) and the **contrast stages carry visibility** (remove the three modules and CLAHE together and real-fog edge F1 falls 0.612 to 0.519; removing only the three modules leaves 0.664 because CLAHE then does the contrast work). Neither replaces the other. Each module on its own pushes edge counts up, but **in the combined pipeline most of them reduce SSIM and detection recall slightly**; that is the price of visibility, and why the app offers presets from *Fog removal only* to *Maximum detail*.

Of the four additions, **only CLAHE changes the result noticeably**: it raises visible-edge gain (real fog +1.14 → +1.43, real streets +0.48 → +0.56) and costs fidelity (real-street SSIM 0.860 → 0.845, real-fog edge F1 0.638 → 0.612). The adaptive dark channel, colour restoration and noise-aware sharpening each move SSIM by 0.003 or less on these scenes, so their effect is not measurable here (colour restoration is meant for the look of real photos, which these scores do not capture). Setting **CLAHE to 0** gives the previous look; the *Fog removal only*, *Course modules only* and *Traffic monitoring* presets already do.

### Fog density: the default over-processes light haze

The Balanced strengths are tuned for dense fog. The app can scale them from the **haze level** of the photo (`haze_level` = 1 - mean dark-channel transmission): `strength = clip(2.1 x haze - 0.35, 0.25, 1)`. The scaling includes the CLAHE strength. With simulated fog at five densities (SSIM, fixed default → adaptive):

| Scene type | Fog 0.4 | Fog 0.8 | Fog 1.6 | Fog 3.2 |
| --- | ---: | ---: | ---: | ---: |
| 7 garden photos | 0.747 → 0.887 | 0.756 → 0.842 | 0.768 → 0.782 | 0.736 → 0.736 |
| Synthetic streets | 0.732 → 0.943 | 0.779 → 0.896 | 0.806 → 0.837 | 0.769 → 0.769 |
| 10 real streets | 0.834 → 0.937 | 0.843 → 0.900 | 0.845 → 0.847 | 0.793 → 0.793 |

The rule was then **fit with one scene type held out** (`python evaluate.py --fit-strength`); on every held-out type it beat the fixed default (scores 1.175 → 1.226, 1.235 → 1.287, 1.284 → 1.333; the per-image best, which needs the clear photo, is 1.267, 1.336, 1.408). The fit on all types now gives steeper constants (2.7 and 0.75) than the shipped 2.1 and 0.35, which were fit on an earlier, smaller run and have not been changed. On the **7 real fog photos**, which were not used for fitting, the gain is small and mixed: SSIM 0.692 → 0.717, PSNR 16.0 → 15.6 dB, edge F1 0.612 → 0.662.

### Are the defaults overfit?

Leave-one-scene-out on real fog (60 random settings of the strengths, ω and colour balance; best on six scenes applied to the seventh): SSIM **0.757** and PSNR 17.5 dB, against 0.703 and 15.8 dB for the shipped default and 0.698 and 13.8 dB for the unprocessed photo. Settings tuned on other scenes beat the shipped default on unseen ones, so **the default is not optimal for real fog**; it was picked to keep the three modules active, and the **Fog removal only** and **Traffic monitoring** presets trade module strength for fidelity. Tuning used 7 scenes of one kind; do not read 0.757 as a benchmark.

### Transmission estimate

On the synthetic streets, whose true depth (and so true transmission `exp(-β·d)`) is known, the dark channel's transmission map (adaptive airlight and t₀ on) has a mean absolute error of 0.094, 0.070, 0.060 and 0.057 at fog 0.8, 1.6, 2.4 and 3.2, a correlation with the true map of 0.91 to 0.96, and an airlight error of 4 to 6 grey levels (`results/transmission.csv`). The **Transmission** view in the app shows this map for any photo.

### Blur estimation

`estimate_blur` measures the width of the strongest edges along **8 directions**, subtracts the width of a sharp photo in quadrature and reports the kind (Gaussian, or motion with its angle) and size. Gaussian and motion blur are told apart by how uneven the excess is across directions; the motion angle comes from a parabola fit to the directional peak. Fog scales an edge's step and gradient alike, so the estimate does not depend on fog density. `wiener_deblur` accepts any angle. Calibration was fit on the real photos only; the synthetic cities are the independent check.

| True blur | Real photos: estimate (px) | kind right | Synthetic cities: estimate (px) | kind right |
| --- | ---: | ---: | ---: | ---: |
| none (fog 0, 1.6, 3) | 0.00 | 100 % | 0.00 | 100 % |
| Gaussian 1 / 2 / 3 / 4 px | 0.84 / 2.00 / 3.16 / 4.14 | 86 / 100 / 100 / 100 % | 1.06 / 1.78 / 2.47 / 3.48 | 100 % |
| Motion 1 / 2 / 3 / 4 px (7 angles) | 0.53 / 2.13 / 2.70 / 3.59 | 47 / 100 / 100 / 100 % | 1.05 / 2.13 / 2.52 / 2.98 | 100 % |

The motion angle is found to **about 3° from 2 px up** (mean error 2.9° real, 3.4° synthetic; 0.3° to 0.5° at 0° and 90°, up to about 7° at 25°/135°). The estimator found no blur in any of the 7 real hazy photos. Using the estimate in the pipeline keeps most of the benefit (fog + 45° motion blur on the garden photos, SSIM 0.635 with no deblur, 0.693 with the estimate, 0.745 with the true blur).

### Video

`dehaze_video.py` runs the pipeline on a video file (`python dehaze_video.py foggy.mp4 clear.mp4 --auto`). Per-frame processing flickers because the airlight, the transmission map, the stretch range and the colour balance move with sensor noise. `TemporalDehazer` smooths those estimates over time, smooths the transmission map only where nothing moved (so cars are not blended into the fog estimate), and averages the sensor noise on still pixels. On the synthetic traffic clip (flicker = mean frame-to-frame change of the output on static pixels, in grey levels):

| Processing | Flicker | PSNR (dB) | SSIM |
| --- | ---: | ---: | ---: |
| Foggy input | 1.54 | 12.4 | 0.747 |
| Every frame on its own | 0.88 | 19.6 | 0.816 |
| **Temporal** | **0.41** | 19.5 | 0.809 |
| Temporal, strength from haze level | 0.35 | 16.9 | 0.827 |

That is about 53 % less flicker than per-frame processing for the same fidelity, at about 0.14 s per 640 x 400 frame on one CPU. These numbers use the video default, which leaves **CLAHE off**: CLAHE re-equalises each frame from that frame's own histogram, so with it on (strength 15) the temporal flicker is 0.63 instead of 0.41. The app's **Video** card uses the current slider settings (limited frame count, H.264 output when `imageio-ffmpeg` is installed), so set CLAHE to 0 there for the steadiest result. Noise-aware sharpening also lowers per-frame flicker (1.19 with the previous defaults, 0.88 now).

### What the numbers mean for the stated impact

- **Traffic monitoring.** In the test, fog hides about a quarter of the vehicles and people from a detector (recall 0.74 on the untouched photo). The Traffic preset restores them to recall 0.91, where the full Balanced pipeline (0.75) does not. The test uses simulated fog on real street photos and a small detector; it is a measured proxy for the impact, not a field trial.
- **Construction inspection.** Visible-edge gain, edge F1 and line recall (building and lane lines) rise strongly after processing, in every scene tested.

**Limitations**

- **No real hazy/clear street pairs.** The real-fog photos are gardens and parks; the real street photos are fogged by simulation with a rough depth guess (farther toward the top of the frame). Real fog on streets has not been measured. The 10 street photos are few and most are photos from one source. The 7 real foggy city photos in `urban_real_foggy/` have no clear references and are not scored.
- **No human study.** Metrics (PSNR, SSIM, Hautière visible-edge gain, edge F1, detector recall) stand in for human judgement and can disagree with it; contrast and edge counts also rise with over-sharpening. Judge the pictures as well. NIQE and BRISQUE are not included (they need trained models); the no-reference visible-edge rows are the offline substitute.
- **The four additions were not tuned.** On these scenes CLAHE trades a little fidelity for visibility and the other three show no measurable effect; their strengths (CLAHE 15, colour restoration 30) were set by hand, not by the parameter search.
- The adaptive strength rule was fit on simulated fog; on real fog its gain is small and mixed, and the shipped constants differ from the latest fit (see above). The Balanced default is not optimal on real fog.
- Blur estimation: blur under about 1 px reads as sharp (a 1 px motion blur is detected in under half of the real photos); large blur is under-read (about 0.5 to 1 px low at 4 px); defocus with a disc-shaped kernel and blur that varies across the frame are not handled. The sharp-photo baseline was measured on photos of about 800 to 1200 px; a resized or heavily compressed photo may read as slightly blurred or sharp.
- Detection uses YOLOv5n (AGPL, not bundled; `detect.py` downloads it on first use). It drives the evaluation and the app's *Applications* view; the app's synthetic pixel-art sample cars are not detected, use a real street photo. The edge and line test uses one detector setting.
- **Failure cases.** Very thick fog with no sky or distant structure gives the dark channel little to measure and the result stays flat; large bright regions (white walls, snow, headlights) are mistaken for haze and darkened; night scenes and coloured fog (yellow smog) break the gray-world and airlight assumptions; strong gamma and *Distant detail* raise noise in dark, far areas. On real fog the default's SSIM sits below dark-channel-only and CLAHE-only (see the methods chart): the brighter, higher-contrast look is bought with fidelity to the clear photo. Gamma 1.15 to 1.2 gives back about 0.6 dB PSNR.
- The temporal video filter assumes a fixed camera; a moving camera makes the motion mask cover the picture, and the filter then falls back to near per-frame behaviour. The video test is a synthetic clip.
- The live demo at the link above shows an older version until it is redeployed from this code.

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
python get_street_photos.py         # download the 8 real street test photos (evaluation only)
pip install -r requirements-eval.txt   # onnxruntime for the vehicle-detection test
python evaluate.py                  # re-run the evaluation, writes ./results (about 5 min)
python evaluate.py --fit-strength   # also re-fit the haze-adaptive strength rule with held-out checks
python evaluate.py --export-urban   # also rewrite the synthetic city test set in assets/
python dehaze_video.py foggy.mp4 clear.mp4 --auto --auto-blur   # dehaze a video
python test_core.py   # quick checks of the image-processing code
```

**Optional:** to open iPhone HEIC/HEIF photos, run `pip install pillow-heif`.

## Project structure

```
Dehaze-App/
├── dehaze_app.py          # Streamlit app (layout, controls, views, batch)
├── dehaze_core.py         # Image processing: stages, fog/blur simulation, blur estimation, metrics (no Streamlit)
├── dehaze_video.py        # Temporal dehazing of video (less flicker) + CLI
├── detect.py              # YOLOv5n vehicle/pedestrian detection (evaluation and the app's Applications view)
├── get_street_photos.py   # Downloads the real street test photos
├── evaluate.py            # Ablation, street, detector, fog-density, blur, video experiments, SVG charts -> results/
├── test_core.py           # Checks for the core
├── results/               # Evaluation output: CSV files, summary.md, comparison images
├── requirements.txt       # Python dependencies (app)
├── requirements-eval.txt  # Extra for the vehicle-detection test (onnxruntime)
├── .streamlit/
│   └── config.toml        # Streamlit theme/config
└── assets/
    ├── cars/              # Pixel-art car sprites (header strip)
    ├── clouds/            # Pixel-art cloud sprites (header strip)
    ├── test images/       # Sample hazy photos to upload and try
    │   ├── urban_real_foggy/  # 7 real foggy or smoggy city photos (no clear references)
    │   ├── urban_real_clear/  # 10 real street photos (download with get_street_photos.py)
    │   └── urban_synthetic/   # 6 synthetic city streets: clear, hazy (β 1.6) and true depth map
    └── Preview/           # Demo recording (preview.gif + full .mp4 recording)
```

## Try it with the sample images

`assets/test images/` includes ready-to-use hazy photos. Drop any of them on the upload box:

- `city_smog/` – smoggy city streets and a foggy park
- `outdoor_fog_with_clear_reference/` – hazy scenes, each paired with a clear reference photo; `evaluate.py` measures them against their references
- `urban_real_foggy/` – real foggy or smoggy city photos (they open as already foggy)
- `urban_real_clear/` – clear real street photos; the app adds simulated fog on them, so PSNR and SSIM against the original work
- `urban_synthetic/` – synthetic city streets with a clear version and a true depth map (`_hazy`, `_clear_reference`, `_depth`); `evaluate.py` uses the depth to simulate fog at other densities

## Tech stack

[Streamlit](https://streamlit.io/) · [OpenCV](https://opencv.org/) (headless) · [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) · [Pillow](https://python-pillow.org/) · imageio-ffmpeg (video output) · onnxruntime + YOLOv5n (evaluation only)

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

## What changed in this version

- **New views**: *Modules* (FFT spectra and filter response, histogram clip points, smoothing and sharpening maps), *Compare* (baselines against the pipeline) and *Applications* (traffic detection, construction edge and line overlay), plus before/after PNG and measurements CSV downloads.
- **Better distant detail**: the far-detail boost is noise-gated; default gamma 1.25 and distant detail 25; the stage-by-stage view shows the original on the left.
- **Batch and video views**: the batch results can be shown as a table, all images, side by side or with a slider; a processed video plays side by side, dehazed or original.
- **Charts**: `evaluate.py` writes SVG charts for baselines, ablation, fog-density sweeps and runtime.
- **Fix**: the video path measured colour-balance gains before colour restoration, so with smoothing off it differed from the still pipeline.

- **CLAHE (stage 03b)**, **adaptive dark channel** (airlight and t₀), **noise-aware sharpening** and **colour restoration**, with the **Transmission** view and no-reference visibility rows (visible-edge gain, gradient ratio, newly clipped pixels).
- **Fix**: an image enlarged with its hover button is no longer drawn under the Enhancement Pipeline card.
- **Fix**: a photo without `_hazy` in its file name is no longer treated as its own clear reference in the sample list.
- **Evaluation re-run** with the new defaults against the previous ones (see above): the additions raise visibility and lower fidelity slightly, almost all of it from CLAHE.
- **Video**: the default settings leave CLAHE off, because it flickers.
- **Test photos**: 7 real foggy city photos and 10 clear real street photos (kept with their original licences: sources github.com/matterport/Mask_RCNN, ultralytics/assets, Utkarsh-Deshmukh/Single-Image-Dehazing-Python and He-Zhang/image_dehaze).
