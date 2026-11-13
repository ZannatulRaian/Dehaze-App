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
| Adaptive smoothing and sharpening | Stages 05 and 06: edge-preserving smoothing, then capped, noise-aware unsharp masking |
| Simulate foggy images | Input stage: fog from the atmospheric scattering model, plus optional Gaussian or motion blur |
| Or use real ones | Upload any photo (tick *Photo is already foggy*); 20 real foggy city photos, 7 outdoor fog photos with clear references and 10 clear street photos are bundled, and `assets/test images/urban_synthetic/` adds synthetic city streets with clear versions and true depth maps |
| Make urban structures more visible | **Edges** view, edge-pixel and gradient measurements, Canny edge and Hough line counts, before/after histograms |
| Blurred or foggy view | Fog removal (dark channel prior) and Wiener deblurring with automatic blur-size estimation (stage 02) |
| Traffic monitoring, construction inspection | **Traffic monitoring** preset, **Applications** views (vehicle detection, edge and line overlay), **Batch Processing** card (many photos, ZIP download) and **Video** card (temporal smoothing) |
| Show that it works on real photos | **Evaluation & Project Health** panel: no-reference evaluation of the bundled real foggy city photos and your own uploads, with CSV, contact-sheet and human-review downloads |

The Dark Channel Prior recovery also includes a soft bright-neutral-region safeguard: it blends back some original pixels in bright, low-saturation areas where the dark-channel assumption is unreliable (e.g. white facades and headlights). This reduces over-darkening but can intentionally leave some bright haze; it is not semantic object recognition.

The **Course modules only** switch runs just the three project modules, so their effect can be shown on its own. The **Dark Channel Prior** and the optional **Wiener deblur** are additions that handle fog and blur, which the three modules alone cannot remove. **CLAHE** (local contrast) and **colour restoration** are further additions that work on what is left after the fog is gone.

## Live demo

👉 **https://dehazeapp.streamlit.app/** (this may show an older version until it is redeployed from this code)

## What is in the app

Everything is on the main page (no sidebar), under the pixel-art header strip.

**Preview card** (left)
- Upload a photo by dropping it on the box. **Change Image** starts over.
- **View** switch:
  - **Side by side** — original and dehazed.
  - **Slider** — one image with a draggable divider between original (left) and dehazed (right).
  - **Original** and **Dehazed** — one image at a time.
  - **Stages** — steps through the pipeline in the order it runs. *Cumulative* shows the image before and after the selected stage; *This stage only* shows the original next to the original with just that stage applied.
  - **Edges** — the Sobel gradient of the original and the dehazed image on one shared scale, to show how much more of the buildings and roads becomes visible.
  - **Transmission** — the original next to the dark channel's estimated transmission map as a colour map (dark = thick haze or far away, bright = clear or near).
  - **Modules** — the three course modules made visible: the FFT spectrum of the original and the dehazed image on one scale, their mean spectrum by frequency next to the frequency filter's gain curve, the luminance histogram before the stretch with its clip points marked and after it, and two maps showing where the adaptive smoothing changes pixels and how strongly the adaptive sharpening acts.
  - **Compare** — the full pipeline next to single-technique baselines (histogram equalisation, CLAHE only, gamma only, plain unsharp mask, dark channel prior only), each with its contrast and edge share, and PSNR and SSIM when a clear reference exists.
  - **Applications** — *Traffic monitoring*: vehicles and people found by a YOLOv5n detector before and after, drawn as boxes (red cars and trucks, orange motorcycles and people), with a confidence slider and a *tiles* switch that also searches overlapping zoomed-in tiles for small, distant vehicles and motorcycles (needs `onnxruntime`, which is in `requirements.txt`; the model file is downloaded on first use if `models/yolov5n.onnx` is missing, and without it the tab says so). *Construction inspection*: Canny edges (red) and Hough lines (green) before and after, with the counts. Both tabs have **Download dehazed**, **Download original** and **Download side by side** buttons that save the annotated images (boxes, or edge and line overlay) as PNG.
- **Measurements**: contrast, entropy, mean gradient, edge pixels, processing time, how many edges a standard Canny detector finds and how many building (vertical) and lane (oblique) lines a Hough line detector finds, and, when fog or blur is simulated on a clear photo (that photo is the reference), PSNR, SSIM, edge F1 and line recall against it, plus brightness and R, G, B histogram comparisons. Three no-reference rows compare the result with the foggy input: visible-edge gain e, gradient ratio and newly clipped pixels.
- Under the measurements: **Download before/after PNG** and **Download measurements CSV**.
- The card stays in view while scrolling through the pipeline controls. An image enlarged with its hover button covers the whole window.

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

**Evaluation & Project Health** (bottom, collapsed): optional tools for report evidence and installation checks, so nothing has to be run in a terminal.
- **1. Real-world fog validation** — runs four modes (Dark Channel Prior only, Traffic monitoring, haze-adaptive Balanced, Course modules only) on the 20 bundled real foggy city photos plus any photos you upload, at a chosen maximum size. It shows per-image and average no-reference indicators (visible-edge gain, gradient ratio, newly clipped pixels, contrast, entropy, edge share, colourfulness, runtime) and offers downloads of the metrics CSV, a comparison contact sheet and a blank human-review sheet.
- **2. Run built-in system checks** — six checks with their observed output: core import, output shape and dtype, disabled stages leave pixels unchanged, identity SSIM and PSNR, the bright-region safeguard, and that the bundled real-fog set is readable.

## How it works

The pipeline (in `dehaze_core.py`) runs in this order; the numbers in brackets are the stage numbers shown in the app:

| Step | Stage | What it does |
| ---- | ----- | ------------ |
| 1 | **Dark Channel Prior** [01] (on by default) | Estimates the airlight and a per-pixel transmission map, refined with a **guided filter** so it follows real edges, then recovers the haze-free scene `J = (I − A) / t + A`. With *adaptive* on, the airlight is the mean of the brightest fifth of the haze candidates and the floor t₀ is scaled to the measured haze (`t₀ · clip(1.5 − 1.6·haze, 0.6, 1.4)`) |
| 2 | **Deblur (Wiener)** [02] (off at 0) | Wiener deconvolution of luminance in the Fourier domain for a Gaussian or linear motion blur (any angle) with a configured size or a size measured by `estimate_blur`; mirror padding and a noise-damping term limit ringing |
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

All numbers in this section come from `python evaluate.py` on this version of the code (CSV files, `summary.md` and the charts are in `results/`). Charts (SVG, regenerate with `python evaluate.py --charts`, about 15 s): [methods against simple baselines (SSIM, edge F1)](results/figures/chart_methods_ssim_f1.svg), [PSNR](results/figures/chart_methods_psnr.svg), [ablation](results/figures/chart_ablation.svg), fog-density sweeps ([garden photos](results/figures/chart_sweep_7_garden_photos.svg), [synthetic streets](results/figures/chart_sweep_synthetic_streets.svg), [real streets](results/figures/chart_sweep_10_real_streets.svg)) and [runtime against image size](results/figures/chart_runtime.svg). The sweep charts read `results/strength_sweep.csv`, which a full `python evaluate.py` run writes.

The first version of the defaults over-processed: on the seven real foggy test scenes it scored **worse than doing nothing** on structural similarity (SSIM 0.58 against 0.69 for the untouched photo). The defaults were then chosen by a random parameter search on four of the scenes, with the three course modules required to stay active. The same search with only the three modules allowed gives the **Course modules only** preset. The other presets are hand-set around the default.

Four additions were made later (CLAHE, the adaptive dark channel, noise-aware sharpening, colour restoration). They were **not** part of the parameter search; CLAHE (strength 15) and colour restoration (30) were set by hand. The ablation below switches the first three off one at a time and together, so what they cost can be read directly; noise-aware sharpening has no switch and is always on.

`python evaluate.py` re-runs everything and writes `results/` (CSV files, `summary.md`, comparison images; about 7 minutes, about 10 with `--fit-strength`). Every method is measured against the clear photo of each scene. There are four kinds of test data:

| Test data | What it is | What it can and cannot show |
| --- | --- | --- |
| **Real fog** | 7 hazy garden and park photos, each with a clear reference | Real fog, but not city streets |
| **Real streets** | 10 clear photos of real streets (`python get_street_photos.py` downloads 9: 8 from the Mask R-CNN sample images and a bus scene from the ultralytics sample images; one more is bundled), fogged with the scattering model and a rough depth guess | Real street content (cars, people, buildings), but simulated fog |
| **Synthetic streets** | 6 renders in `assets/test images/urban_synthetic/` (buildings, cars, lane lines) with a clear version and the **true depth map** | Exact ground truth; clean flat colours flatter edge detection |
| **Synthetic traffic clip** | 30 frames, fixed camera, 3 cars, fog from the true depth, sensor noise | Video behaviour |

**No real hazy/clear photo pairs of city streets are bundled**, because none were available. `assets/test images/urban_real_foggy/` has 20 real foggy or smoggy city photos, but without clear references, so they can be judged by eye and by the no-reference rows (visible-edge gain e, gradient ratio), not by PSNR or SSIM (see *Real-world validation* below). To use real pairs, put `name_hazy.jpg` and `name_clear_reference.jpg` in `assets/test images/urban_with_clear_reference/`; `evaluate.py` adds a section for them.

### Visibility and fidelity (SSIM and PSNR against the clear photo)

| Method | Real fog: SSIM | PSNR | Real streets, fog 1.6: SSIM | PSNR | Synthetic streets: SSIM | PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input | 0.674 | 13.7 | 0.647 | 10.9 | 0.754 | 12.2 |
| The three course modules together | 0.685 | 13.7 | 0.755 | 14.3 | 0.696 | 12.2 |
| Dark channel prior only | 0.774 | 16.9 | 0.917 | 21.1 | 0.871 | 21.7 |
| Full pipeline (Balanced default) | 0.693 | 15.2 | 0.810 | 17.9 | 0.796 | 17.2 |
| Full pipeline, strength set from the haze level | 0.719 | 14.6 | 0.817 | 17.3 | 0.812 | 14.8 |
| Traffic monitoring preset | 0.733 | 14.9 | 0.922 | 21.0 | 0.865 | 19.9 |

Visibility without a clear photo (**visible-edge gain e** and **gradient ratio r**, Hautière et al., against the foggy input; Balanced default): real fog e +1.57 and r 3.96; real streets e +0.63 and r 3.89; synthetic streets e +1.42 (r is not meaningful there: the flat-colour renders have almost no gradient to start with). Against dark channel prior only, the Balanced pipeline gains visibility (real fog e +0.85 → +1.57, real streets +0.48 → +0.63) and loses fidelity (real-street SSIM 0.917 → 0.810).

### Impact test: what standard detectors find before and after

Canny (fixed 50/150) and a Hough line detector are run on each result and compared with what they find in the clear photo. **Edge F1** (2 px tolerance) and **line recall** are the numbers to read; raw counts rise with texture and noise too. On the real streets a **YOLOv5n** detector (`detect.py`, weights downloaded on first use, AGPL licence) also counts vehicles and people; *detection recall* is the share of what it finds in the clear photo that it still finds (IoU 0.5).

| Method | Real fog: edge F1 | line recall | Real streets: edge F1 | line recall | vehicles/people found (clear photo: 11.3) | detection recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input | 0.155 | 0.039 | 0.450 | 0.085 | 9.3 | 0.731 |
| Histogram stretching only | 0.642 | 0.443 | 0.818 | 0.404 | 10.6 | 0.848 |
| The three course modules together | 0.661 | 0.520 | 0.846 | 0.527 | 8.3 | 0.692 |
| Dark channel prior only | 0.462 | 0.166 | 0.895 | 0.480 | 10.9 | 0.890 |
| Full pipeline (Balanced default) | 0.597 | 0.584 | 0.807 | 0.634 | 9.5 | 0.761 |
| Full pipeline, strength set from the haze level | 0.653 | 0.548 | 0.848 | 0.650 | 9.5 | 0.774 |
| **Traffic monitoring preset** | 0.660 | 0.402 | **0.928** | 0.606 | 11.1 | 0.886 |
| Traffic monitoring preset, strength set from the haze level | 0.622 | 0.373 | 0.914 | 0.545 | **11.3** | **0.900** |

Findings:

- On real fog the standard detectors find almost nothing in the photo (0.4 % edge pixels, 6 lines) and a lot after processing; edge F1 improves in every scene for every method (7 of 7).
- **The full pipeline's strong sharpening and frequency stages find more edges and lines but *lose* vehicles**: detection recall 0.761 against 0.731 for the untouched photo and 0.890 for the dark channel alone, because over-sharpened and over-stretched cars no longer look like cars to the detector. For traffic monitoring this is the result that matters, so there is a **Traffic monitoring** preset (fog removal + histogram stretching only, strength set from the haze level in the app). It gives the best edge F1 (0.928) on the real streets, and its detection recall (0.886; 0.900 with the strength set from the haze level) matches or beats the dark channel alone (0.890) and every other method tested.
- The preset's gain was checked on two disjoint halves of the street photos at three fog densities and held in each, for an earlier version of the defaults; that check was not repeated for this run.
- Building (near-vertical) and lane (oblique) line counts are split out in `summary.md`. The three course modules raise lane-line counts from 8 to 96 on the real streets.
- With only 10 street photos, differences of a few hundredths in detection recall are within noise.

### Do the three course modules matter next to the dark channel prior?

Each row removes one part of the Balanced pipeline (`summary.md`, section 6):

| Pipeline | Real fog SSIM / F1 | Real streets SSIM / F1 | Detection recall |
| --- | ---: | ---: | ---: |
| Full pipeline | 0.693 / 0.597 | 0.810 / 0.807 | 0.761 |
| without histogram stretching | 0.754 / 0.642 | 0.809 / 0.822 | 0.781 |
| without frequency-domain enhancement | 0.732 / 0.630 | 0.853 / 0.854 | 0.832 |
| without adaptive smoothing + sharpening | 0.691 / 0.599 | 0.865 / 0.842 | 0.814 |
| without all three modules | 0.801 / 0.696 | 0.885 / 0.918 | 0.866 |
| without the three modules **and CLAHE** | 0.799 / 0.610 | 0.890 / 0.925 | 0.847 |
| without the dark channel prior | 0.671 / 0.660 | **0.739** / 0.837 | 0.750 |
| without CLAHE | 0.714 / 0.621 | 0.832 / 0.843 | 0.761 |
| without colour restoration | 0.695 / 0.599 | 0.810 / 0.807 | 0.771 |
| without the adaptive dark channel | 0.694 / 0.599 | 0.814 / 0.813 | 0.761 |
| without CLAHE, colour restoration and the adaptive dark channel | 0.716 / 0.624 | 0.836 / 0.848 | 0.751 |

The honest reading: the **dark channel prior carries fidelity** (remove it and street SSIM falls 0.810 → 0.739, synthetic-street SSIM 0.796 → 0.714). The **three course modules carry visibility, and pay for it in fidelity**: on the real photos, removing them *raises* SSIM (real fog 0.693 → 0.801, streets 0.810 → 0.885), edge F1 and detection recall (0.761 → 0.866). Their value shows where the fog is already removed but the picture is flat: on the clean synthetic streets the dark channel alone adds almost no visible edges (e +0.04) and the full pipeline adds +1.42, and on real fog a single module, histogram stretching, lifts edge F1 from 0.155 to 0.642 and line recall from 0.039 to 0.443. In the combined pipeline on real photos the dark channel and CLAHE already do much of that contrast work, so the modules mostly trade fidelity for visibility. That is why the app offers presets from *Fog removal only* to *Maximum detail*.

Of the additions, **only CLAHE changes the result noticeably**: removing it raises SSIM (real fog 0.693 → 0.714, real streets 0.810 → 0.832) and edge F1 against the clear photo (0.597 → 0.621, 0.807 → 0.843), and lowers visible-edge gain (real fog +1.57 → +1.30, real streets +0.63 → +0.55). Colour restoration and the adaptive dark channel each move SSIM by 0.004 or less on these scenes, so their effect is not measurable here (colour restoration is meant for the look of real photos, which these scores do not capture). Setting **CLAHE to 0** removes most of the fidelity cost; the *Fog removal only*, *Course modules only* and *Traffic monitoring* presets already do.

### Fog density: the default over-processes light haze

The Balanced strengths are tuned for dense fog. The app can scale them from the **haze level** of the photo (`haze_level` = 1 - mean dark-channel transmission): `strength = clip(2.1 x haze - 0.35, 0.25, 1)`. The scaling includes the CLAHE strength. With simulated fog at four densities (SSIM, fixed default → adaptive):

| Scene type | Fog 0.4 | Fog 0.8 | Fog 1.6 | Fog 3.2 |
| --- | ---: | ---: | ---: | ---: |
| 7 garden photos | 0.736 → 0.870 | 0.740 → 0.824 | 0.743 → 0.759 | 0.681 → 0.681 |
| Synthetic streets | 0.825 → 0.929 | 0.826 → 0.885 | 0.796 → 0.812 | 0.718 → 0.714 |
| 10 real streets | 0.821 → 0.911 | 0.818 → 0.872 | 0.810 → 0.817 | 0.746 → 0.746 |

The rule was then **fit with one scene type held out** (`python evaluate.py --fit-strength`); on every held-out type it beat the fixed default (scores 1.210 → 1.252, 1.215 → 1.301, 1.294 → 1.337; the per-image best, which needs the clear photo, is 1.283, 1.342, 1.418). The fit on all types gives steeper constants (2.4 and 0.45) than the shipped 2.1 and 0.35, which were fit on an earlier, smaller run and have not been changed. On the **7 real fog photos**, which were not used for fitting, the gain is small and mixed: SSIM 0.693 → 0.719, PSNR 15.2 → 14.6 dB, edge F1 0.597 → 0.653.

### Are the defaults overfit?

Leave-one-scene-out on real fog (60 random settings of the strengths, ω and colour balance; best on six scenes applied to the seventh): SSIM **0.739** and PSNR 15.2 dB, against 0.711 and 15.4 dB for the shipped default and 0.698 and 13.8 dB for the unprocessed photo (the best setting found for each scene separately reaches 0.780 and 16.5 dB). Settings tuned on other scenes beat the shipped default on SSIM for unseen ones, with no PSNR gain, so **the default is not optimal for real fog**; it was picked to keep the three modules active, and the **Fog removal only** and **Traffic monitoring** presets trade module strength for fidelity. Tuning used 7 scenes of one kind; do not read 0.739 as a benchmark.

### Transmission estimate

On the synthetic streets, whose true depth (and so true transmission `exp(-β·d)`) is known, the dark channel's transmission map (adaptive airlight and t₀ on) has a mean absolute error of 0.094, 0.070, 0.060 and 0.056 at fog 0.8, 1.6, 2.4 and 3.2, a correlation with the true map of 0.91 to 0.96, and an airlight error of 4 to 6 grey levels (`results/transmission.csv`). The **Transmission** view in the app shows this map for any photo.

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
| Every frame on its own | 1.45 | 16.9 | 0.787 |
| **Temporal** | **0.72** | 17.0 | 0.793 |
| Temporal, strength from haze level | 0.61 | 14.5 | 0.807 |
| Temporal, **CLAHE off** | **0.51** | 17.4 | 0.807 |
| Temporal, CLAHE off, strength from haze level | 0.38 | 14.7 | 0.817 |

Temporal processing roughly halves the flicker of per-frame processing for the same fidelity, at about 0.17 s per 640 x 400 frame on one CPU. The video code uses the Balanced default, which has **CLAHE on**: CLAHE re-equalises each frame from that frame's own histogram, so it is the main source of the remaining flicker (0.72 with it, 0.51 without). The app's **Video** card uses the current slider settings (limited frame count, H.264 output when `imageio-ffmpeg` is installed), so set CLAHE to 0 there for the steadiest result.

### Real-world validation (unpaired urban fog)

The repository includes **20 real fog/smog urban photographs** in `assets/test images/urban_real_foggy/`. They do not have aligned clear-reference images, so the evaluation intentionally does **not** report PSNR, SSIM, reference edge-F1, or ground-truth detection recall for them. Run it in the app (**Evaluation & Project Health**) or from the terminal:

```bash
python evaluate.py --real-world-only --max-dim 800
```

It compares four modes on every readable image and writes `results/real_world_validation.md` (aggregate results and interpretation guidance), `results/real_world_unpaired.csv` (per-image, per-method indicators and processing time), `results/human_review_template.csv` (blank capture metadata and independent visual-review fields) and `results/figures/real_world_unpaired_contact_sheet.jpg` (originals beside haze-adaptive outputs). Mean over the 20 photos:

| Method | Visible-edge gain e | Gradient ratio r | Newly clipped pixels |
| --- | ---: | ---: | ---: |
| Dark Channel Prior only | +0.401 | 1.56 | 0.07 % |
| Traffic monitoring | +0.770 | 2.06 | 0.84 % |
| Balanced (haze-adaptive) | +0.721 | 2.75 | 0.55 % |
| Course modules only | +0.519 | 2.53 | 1.20 % |

The indicators also include contrast, entropy, edge-pixel share and colourfulness. **These are diagnostic indicators, not ground-truth quality scores**: an increase can also be caused by noise or over-sharpening. Complete the human-review template and collect additional local fog footage before making strong deployment claims.

### What the numbers mean for the stated impact

- **Traffic monitoring.** In the test, fog hides about a quarter of the vehicles and people from a detector (recall 0.73 on the untouched photo). The Traffic preset restores them to recall 0.89 to 0.90, where the full Balanced pipeline (0.76) does not. The test uses simulated fog on real street photos and a small detector; it is a measured proxy for the impact, not a field trial.
- **Construction inspection.** Visible-edge gain, edge F1 and line recall (building and lane lines) rise strongly after processing, in every scene tested.

**Limitations**

- **No real hazy/clear street pairs.** The real-fog photos are gardens and parks; the real street photos are fogged by simulation with a rough depth guess (farther toward the top of the frame). Real fog on streets has not been measured. The 10 street photos are few and most are photos from one source. The 20 real foggy city photos in `urban_real_foggy/` have no clear references and are evaluated separately with unpaired indicators.
- **No completed human study.** A blank independent-review template is generated for the 20 real urban fog photos, but it must be completed by reviewers. PSNR, SSIM, visible-edge gain, edge F1 and detector recall can disagree with human judgement; contrast and edge counts can also rise with over-sharpening. NIQE and BRISQUE are not included; the supplied no-reference indicators are diagnostic proxies, not substitutes for those learned metrics or a human study.
- **The four additions were not tuned.** On these scenes CLAHE trades fidelity for visibility and the others show no measurable effect; their strengths (CLAHE 15, colour restoration 30) were set by hand, not by the parameter search.
- The adaptive strength rule was fit on simulated fog; on real fog its gain is small and mixed, and the shipped constants differ from the latest fit (see above). The Balanced default is not optimal on real fog.
- Blur estimation: blur under about 1 px reads as sharp (a 1 px motion blur is detected in under half of the real photos); large blur is under-read (about 0.5 to 1 px low at 4 px); defocus with a disc-shaped kernel and blur that varies across the frame are not handled. The sharp-photo baseline was measured on photos of about 800 to 1200 px; a resized or heavily compressed photo may read as slightly blurred or sharp.
- Detection uses YOLOv5n (AGPL, not committed to the repository; `detect.py` downloads it on first use). It drives the evaluation and the app's *Applications* view; the app's synthetic pixel-art sample cars are not detected, use a real street photo. The edge and line test uses one detector setting.
- **Failure cases.** Very thick fog with no sky or distant structure gives the dark channel little to measure and the result stays flat; the bright-neutral safeguard reduces but cannot eliminate errors on white walls, snow and headlights; night scenes and coloured fog (yellow smog) break the gray-world and airlight assumptions; strong gamma and *Distant detail* raise noise in dark, far areas. On real fog the default's SSIM sits below dark-channel-only (see the methods chart): the brighter, higher-contrast look is bought with fidelity to the clear photo. Gamma 1.15 to 1.2 gives back some PSNR.
- The temporal video filter assumes a fixed camera; a moving camera makes the motion mask cover the picture, and the filter then falls back to near per-frame behaviour. The video test is a synthetic clip.
- The live demo at the link above shows an older version until it is redeployed from this code.

### Field-validation protocol

1. Collect images/videos from multiple urban locations under light, moderate and dense fog, including daytime and nighttime where safe.
2. Record date/time, approximate location, camera, weather/visibility description, lighting and source/licence for each capture.
3. Keep original files unchanged; do not use the same scene in both parameter tuning and final testing.
4. Where possible, capture aligned clear-weather references from the same fixed camera and viewpoint. Use PSNR/SSIM only for geometrically aligned pairs.
5. For traffic monitoring, annotate vehicles and pedestrians manually and report precision, recall and mAP against those annotations. Agreement with a model's own clear-image detections is not ground truth.
6. Ask at least two reviewers to score visibility, artifacts and colour naturalness independently; report the mean and reviewer agreement.

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

The app opens at `http://localhost:8501`. **Normal startup requires only these two terminal commands:**

```bash
pip install -r requirements.txt
streamlit run dehaze_app.py
```

Keep the terminal window open while using the browser app. `requirements.txt` includes `onnxruntime`, so vehicle detection works without further installs (the YOLOv5n model file downloads on first use if it is not already in `models/`). The page includes **Evaluation & Project Health** at the bottom: run the real-world fog benchmark, upload foggy urban photographs, inspect per-image and average metrics, download the CSV, contact sheet and human-review template, and run built-in system checks directly in the app. No separate evaluation or test command is required.

The additional scripts below are optional developer/maintenance tools only; they are not part of the normal project workflow:

- `python evaluate.py` — full reference-based research benchmark.
- `python evaluate.py --fit-strength` — refit the haze-adaptive strength rule.
- `python evaluate.py --real-world-only --max-dim 800` — only the unpaired real urban fog evaluation.
- `python evaluate.py --export-urban` — regenerate synthetic urban scenes.
- `python get_street_photos.py` — download optional street test photos.
- `python test_core.py` / `python test_real_world.py` — command-line test suites (the app has its own browser-based system checks).

## Project structure

```
Dehaze-App/
├── dehaze_app.py          # Streamlit app (layout, controls, views, batch, evaluation panel)
├── dehaze_core.py         # Image processing: stages, fog/blur simulation, blur estimation, metrics (no Streamlit)
├── dehaze_video.py        # Temporal dehazing of video (less flicker) + CLI
├── detect.py              # YOLOv5n vehicle/pedestrian detection (Applications views and evaluation)
├── get_street_photos.py   # Downloads the real street test photos
├── evaluate.py            # Ablation, street, detector, fog-density, blur, video, unpaired real-world experiments, SVG charts -> results/
├── test_core.py           # Checks for the core
├── test_real_world.py     # Smoke tests for the unpaired real-world evaluation
├── requirements.txt       # Python dependencies (app, including onnxruntime)
├── models/                # yolov5n.onnx (downloaded on first use; not committed)
├── results/               # Evaluation output: CSV files, summary.md, real_world_validation.md, comparison images, figures/
├── .streamlit/
│   └── config.toml        # Streamlit theme/config
└── assets/
    ├── cars/              # Pixel-art car sprites (header strip)
    ├── clouds/            # Pixel-art cloud sprites (header strip)
    ├── test images/       # Sample hazy photos to upload and try
    │   ├── city_smog/                         # smoggy city streets and a foggy park
    │   ├── outdoor_fog_with_clear_reference/  # 7 hazy scenes, each with a clear reference
    │   ├── urban_real_foggy/                  # 20 real foggy or smoggy city photos (no clear references)
    │   ├── urban_real_clear/                  # 10 real street photos (download with get_street_photos.py)
    │   └── urban_synthetic/                   # 6 synthetic city streets: clear, hazy (β 1.6) and true depth map
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

[Streamlit](https://streamlit.io/) · [OpenCV](https://opencv.org/) (headless) · [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) · [Pillow](https://python-pillow.org/) · imageio-ffmpeg (video output) · onnxruntime + YOLOv5n (vehicle detection in the Applications views and the evaluation)

## References

- K. He, J. Sun, X. Tang, *Single Image Haze Removal Using Dark Channel Prior*, CVPR 2009 (IEEE TPAMI 2011)
- K. He, J. Sun, X. Tang, *Guided Image Filtering*, ECCV 2010 (IEEE TPAMI 2013)
- S. G. Narasimhan, S. K. Nayar, *Vision and the Atmosphere*, IJCV 2002
- H. Koschmieder, *Theorie der horizontalen Sichtweite*, 1924
- R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018
- C. Tomasi, R. Manduchi, *Bilateral Filtering for Gray and Color Images*, ICCV 1998
- Z. Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE TIP 2004
- N. Hautière, J.-P. Tarel, D. Aubert, É. Dumont, *Blind Contrast Enhancement Assessment by Gradient Ratioing at Visible Edges*, Image Analysis & Stereology, 2011

## Author

 [ZannatulRaian](https://github.com/ZannatulRaian)

## What changed in this version

- **Evaluation & Project Health panel** in the app: real-world fog validation on the bundled and uploaded photos (per-image and average no-reference indicators, metrics CSV, contact sheet, blank human-review sheet) and six built-in system checks, so no terminal command is needed.
- **Unpaired real-world validation** of 20 real foggy city photos (`python evaluate.py --real-world-only`), `test_real_world.py`, and a field-validation protocol; the report states what these photos can and cannot show.
- **Bright-neutral safeguard** in the dark channel recovery, to reduce over-darkening of white facades and headlights.
- **Evaluation re-run on this code and the README numbers regenerated from it.** `evaluate.py` now also ablates CLAHE, colour restoration and the adaptive dark channel (alone and together) and measures the video filter with CLAHE off, so every claim above can be reproduced. The sweep label now counts the real street photos correctly (10, chart `chart_sweep_10_real_streets.svg`).
- **Download annotated images**: the Applications tabs (single photo, batch and video frames) can save the detection image or the edge and line overlay as PNG, dehazed, original or side by side. The explanatory caption under the tile switch is removed.
- **Detector fix**: `detect.py` scored only each box's single best COCO class and ran one class-agnostic suppression, so a motorcycle could be dropped or hidden under its rider's box. It now scores every wanted class and suppresses only within look-alike groups (car/bus/truck, bicycle/motorcycle). More motorcycles are found; detection numbers above come from this version. Small, distant riders in dense fog are still often missed or called a car or person: YOLOv5n is the smallest YOLO model.
- **Setup**: `onnxruntime` is part of `requirements.txt`, so the Applications views work after one `pip install`.
- **New views**: *Modules* (FFT spectra and filter response, histogram clip points, smoothing and sharpening maps), *Compare* (baselines against the pipeline) and *Applications* (traffic detection, construction edge and line overlay), plus before/after PNG and measurements CSV downloads.
- **Better distant detail**: the far-detail boost is noise-gated; default gamma 1.25 and distant detail 25; the stage-by-stage view shows the original on the left.
- **Batch and video views**: the batch results can be shown as a table, all images, side by side or with a slider; a processed video plays side by side, dehazed or original.
- **Charts**: `evaluate.py` writes SVG charts for baselines, ablation, fog-density sweeps and runtime.
- **CLAHE (stage 03b)**, **adaptive dark channel** (airlight and t₀), **noise-aware sharpening** and **colour restoration**, with the **Transmission** view and no-reference visibility rows (visible-edge gain, gradient ratio, newly clipped pixels).
- **Fixes**: the video path measured colour-balance gains before colour restoration, so with smoothing off it differed from the still pipeline; an image enlarged with its hover button is no longer drawn under the Enhancement Pipeline card; a photo without `_hazy` in its file name is no longer treated as its own clear reference in the sample list.
- **Test photos**: 20 real foggy city photos and 10 clear real street photos (kept with their original licences: sources github.com/matterport/Mask_RCNN, ultralytics/assets, Utkarsh-Deshmukh/Single-Image-Dehazing-Python and He-Zhang/image_dehaze).
