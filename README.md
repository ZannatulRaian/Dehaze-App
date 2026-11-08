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


| Project requirement                            | Where it is in the app                                                                                                                                                                                                                             |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Frequency-domain enhancement (edge sharpening) | Stage 04: FFT high-frequency emphasis with a Gaussian high-pass                                                                                                                                                                                    |
| Histogram stretching                           | Stage 03: percentile stretch of luminance                                                                                                                                                                                                          |
| Adaptive smoothing and sharpening              | Stages 05 and 06: edge-preserving smoothing, then capped unsharp masking                                                                                                                                                                           |
| Simulate foggy images                          | Input stage: fog from the atmospheric scattering model, plus optional Gaussian or motion blur                                                                                                                                                      |
| Or use real ones                               | Upload any photo (tick*Photo is already foggy*); the bundled outdoor test photos come with clear references that `evaluate.py` uses, and `assets/test images/urban_synthetic/` adds synthetic city streets with clear versions and true depth maps |
| Make urban structures more visible             | **Edges** view, edge-pixel and gradient measurements, Canny edge and Hough line counts, before/after histograms                                                                                                                                    |
| Blurred or foggy view                          | Fog removal (dark channel prior) and Wiener deblurring with automatic blur-size estimation (stage 02)                                                                                                                                              |
| Traffic monitoring, construction inspection    | **Traffic monitoring** preset, **Batch Processing** card (many photos, ZIP download) and **Video** card (temporal smoothing)                                                                                                                       |

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
- **Measurements**: contrast, entropy, mean gradient, edge pixels, processing time, how many edges a standard Canny detector finds and how many building (vertical) and lane (oblique) lines a Hough line detector finds, and, when fog or blur is simulated on a clear photo (that photo is the reference), PSNR, SSIM, edge F1 and line recall against it, plus brightness and R, G, B histogram comparisons.
- The card stays in view while you scroll the pipeline controls.

**Enhancement Pipeline card** (right)

- **00 Input** — *Photo is already foggy*, simulated fog density, simulated blur (Gaussian or motion at any angle), and the **Course modules only** switch.
- **01 Dark Channel Prior** — on/off, haze removed (ω), minimum transmission (t₀), patch size.
- **02 Deblur (Wiener)** — blur to undo (0 = off), blur type, noise damping, or **Detect the blur automatically**, which measures the blur type and size from the photo and shows what it found.
- **03 Histogram Stretch**, **04 Frequency-Domain Filter** (strength and cutoff radius), **05 Adaptive Smoothing**, **06 Adaptive Sharpening**.
- **07 Final Tone** — gamma, saturation and a gray-world colour balance.
- **Preset** (Balanced, Light haze, Dense fog, Maximum detail, Fog removal only, Course modules only, Traffic monitoring), **Auto-Enhance**, **Reset**, **Download Result**.

**Video** card: upload a short clip, choose how many frames to process and whether to smooth over time (less flicker), then download the dehazed mp4.

**Batch Processing** (below): upload several photos, process them with the current settings, see time and contrast per photo, and download everything as a ZIP with a CSV of the results.

**Method** (bottom): the fog model, the approach and the references.

## How it works

The pipeline (in `dehaze_core.py`) runs in this order; the numbers in brackets are the stage numbers shown in the app:


| Step | Stage                                       | What it does                                                                                                                                                                                                       |
| ---- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | **Dark Channel Prior** [01] (on by default) | Estimates the airlight and a per-pixel transmission map, refined with a**guided filter** so it follows real edges, then recovers the haze-free scene `J = (I − A) / t + A`                                        |
| 2    | **Deblur (Wiener)** [02] (off at 0)         | Wiener deconvolution of luminance in the Fourier domain for a Gaussian or linear motion blur (any angle) whose size you set or that`estimate_blur` measures; mirror padding and a noise-damping term limit ringing |
| 3    | **Adaptive Smoothing** [05]                 | Edge-preserving bilateral filter; smooths flat areas and the noise fog removal amplifies, keeps edges                                                                                                              |
| 4    | **Adaptive Sharpening** [06]                | Unsharp mask on luminance with the added detail capped, so strong edges don't overshoot into halos                                                                                                                 |
| 5    | **Histogram Stretch** [03]                  | Linear contrast stretch of luminance, clipping 0–2 % from each end                                                                                                                                                |
| 6    | **Frequency-Domain Filter** [04]            | FFT high-frequency emphasis with a smooth Gaussian high-pass (no ringing) and mirror padding (no border wrap-around)                                                                                               |
| 7    | **Final Tone** [07]                         | Optional gray-world colour balance, saturation and gamma                                                                                                                                                           |

**Design notes**

- Stages 2–6 work on **luminance only** (the Y channel of YCrCb). Processing R, G and B separately shifts hues and paints coloured noise.
- Every stage returns its input untouched when it is switched off or its strength is `0`, so **Reset** switches the four filters and the deblur off and puts the tone controls back to neutral. **Course modules only** switches off the dark channel, deblur and tone stages without touching their settings.
- **Blur estimation** (`estimate_blur`): the width of the strongest edges is measured along 8 directions (the std of the edge's gradient profile, which equals the blur kernel's std). A sharp photo's own edge width is subtracted in quadrature; equal excess in all directions is Gaussian blur, excess that peaks in one direction is motion blur at that angle (box length = 3 x amount). Fog scales an edge's step and its gradient by the same factor, so the estimate does not depend on fog density.
- Images larger than **1200 px** on the long edge are downscaled before processing.
- The image-processing code is in `dehaze_core.py` (no Streamlit), so it can be tested and evaluated on its own.

## Default settings and evaluation

The first version of the defaults over-processed: on the seven real foggy test scenes it scored **worse than doing nothing** on structural similarity (SSIM 0.58 against 0.69 for the untouched photo). The defaults were then chosen by a random parameter search on four of the scenes, with the three course modules required to stay active. The same search with only the three modules allowed gives the **Course modules only** preset. The other presets are hand-set around the default.

`python evaluate.py` re-runs everything and writes `results/` (CSV files, `summary.md`, comparison images; about 5 minutes). Every method is measured against the clear photo of each scene. There are four kinds of test data:


| Test data                  | What it is                                                                                                                                                                             | What it can and cannot show                                      |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **Real fog**               | 7 hazy garden and park photos, each with a clear reference                                                                                                                             | Real fog, but not city streets                                   |
| **Real streets**           | 9 clear photos of real streets (`python get_street_photos.py` downloads 8 from the Mask R-CNN sample images; one is bundled), fogged with the scattering model and a rough depth guess | Real street content (cars, people, buildings), but simulated fog |
| **Synthetic streets**      | 6 renders in`assets/test images/urban_synthetic/` (buildings, cars, lane lines) with a clear version and the **true depth map**                                                        | Exact ground truth; clean flat colours flatter edge detection    |
| **Synthetic traffic clip** | 30 frames, fixed camera, 3 cars, fog from the true depth, sensor noise                                                                                                                 | Video behaviour                                                  |

**No real hazy/clear photo pairs of city streets are bundled**, because none were available. To use some, put `name_hazy.jpg` and `name_clear_reference.jpg` in `assets/test images/urban_with_clear_reference/`; `evaluate.py` adds a section for them.

### Visibility and fidelity (SSIM and PSNR against the clear photo)


| Method                                          | Real fog: SSIM | PSNR | Real streets, fog 1.6: SSIM | PSNR | Synthetic streets: SSIM | PSNR |
| ----------------------------------------------- | -------------: | ---: | --------------------------: | ---: | ----------------------: | ---: |
| Foggy input                                     |          0.674 | 13.7 |                       0.645 | 10.9 |                   0.754 | 12.2 |
| The three course modules together               |          0.684 | 13.7 |                       0.759 | 14.2 |                   0.696 | 12.2 |
| Dark channel prior only                         |          0.769 | 17.0 |                       0.905 | 19.2 |                   0.871 | 21.8 |
| Full pipeline (Balanced default)                |          0.706 | 15.9 |                       0.866 | 18.8 |                   0.793 | 19.4 |
| Full pipeline, strength set from the haze level |          0.718 | 15.5 |                       0.860 | 18.2 |                   0.837 | 17.4 |
| Traffic monitoring preset                       |          0.739 | 15.1 |                       0.918 | 20.0 |                   0.866 | 20.2 |

### Impact test: what standard detectors find before and after

Canny (fixed 50/150) and a Hough line detector are run on each result and compared with what they find in the clear photo. **Edge F1** (2 px tolerance) and **line recall** are the numbers to read; raw counts rise with texture and noise too. On the real streets a **YOLOv5n** detector (`detect.py`, evaluation only, weights downloaded on first use, AGPL licence) also counts vehicles and people; *detection recall* is the share of what it finds in the clear photo that it still finds (IoU 0.5).


| Method                                          | Real fog: edge F1 | line recall | Real streets: edge F1 | line recall | vehicles/people found (clear photo: 11.9) | detection recall |
| ----------------------------------------------- | ----------------: | ----------: | --------------------: | ----------: | ----------------------------------------: | ---------------: |
| Foggy input                                     |             0.155 |       0.039 |                 0.446 |       0.083 |                                       9.8 |            0.714 |
| Histogram stretching only                       |             0.642 |       0.443 |                 0.819 |       0.407 |                                      11.0 |            0.831 |
| The three course modules together               |             0.661 |       0.513 |                 0.854 |       0.527 |                                       8.8 |            0.699 |
| Dark channel prior only                         |             0.441 |       0.150 |                 0.874 |       0.430 |                                      11.2 |            0.881 |
| Full pipeline (Balanced default)                |             0.645 |       0.551 |                 0.889 |       0.652 |                                       9.9 |            0.725 |
| Full pipeline, strength set from the haze level |             0.664 |       0.509 |                 0.900 |       0.607 |                                      10.0 |            0.747 |
| **Traffic monitoring preset**                   |             0.674 |       0.434 |             **0.924** |       0.578 |                                  **11.6** |        **0.895** |

Findings:

- On real fog the standard detectors find almost nothing in the photo (0.4 % edge pixels, 6 lines) and a lot after processing; edge F1 improves in every scene for every method.
- **The full pipeline's strong sharpening and frequency stages find more edges and lines but *lose* vehicles**: detection recall 0.725 against 0.714 for the untouched photo and 0.881 for the dark channel alone, because over-sharpened and over-stretched cars no longer look like cars to the detector. For traffic monitoring this is the result that matters, so there is a **Traffic monitoring** preset (fog removal + histogram stretching only, strength set from the haze level in the app). It gives the best edge F1 and detection recall on the real streets (0.924 and 0.895; the strength-adaptive variant is 0.911 and 0.863).
- The preset's gain was checked on two disjoint halves of the street photos at three fog densities and holds in each.
- Building (near-vertical) and lane (oblique) line counts are split out in `summary.md`. The three course modules raise lane-line counts from 7 to 77 on the real streets.

### Do the three course modules matter next to the dark channel prior?

Each row removes one part of the Balanced pipeline (`summary.md`, section 6):


| Pipeline                                | Real fog SSIM / F1 | Real streets SSIM / F1 | Detection recall |
| --------------------------------------- | -----------------: | ---------------------: | ---------------: |
| Full pipeline                           |      0.706 / 0.645 |          0.866 / 0.889 |            0.725 |
| without histogram stretching            |      0.766 / 0.689 |          0.870 / 0.903 |            0.725 |
| without frequency-domain enhancement    |      0.729 / 0.688 |          0.892 / 0.918 |            0.732 |
| without adaptive smoothing + sharpening |      0.710 / 0.668 |          0.908 / 0.919 |            0.782 |
| without all three modules               |   0.783 /**0.491** |          0.906 / 0.871 |            0.839 |
| without the dark channel prior          |      0.687 / 0.685 |      **0.768** / 0.856 |            0.735 |

The honest reading: the **dark channel prior carries fidelity** (remove it and street SSIM falls 0.866 to 0.768) and the **three modules carry visibility** (remove them and real-fog edge F1 falls 0.645 to 0.491). Neither replaces the other. Each module on its own pushes edge counts up, but **in the combined pipeline most of them reduce SSIM and detection recall slightly**; that is the price of visibility, and why the app offers presets from *Fog removal only* to *Maximum detail*.

### Fog density: the default over-processes light haze

The Balanced strengths are tuned for dense fog. The app can scale them from the **haze level** of the photo (`haze_level` = 1 - mean dark-channel transmission): `strength = clip(2.1 x haze - 0.35, 0.25, 1)`. With simulated fog at five densities (SSIM, fixed default → adaptive):


| Scene type        |        Fog 0.4 |        Fog 0.8 |        Fog 1.6 |        Fog 3.2 |
| ----------------- | -------------: | -------------: | -------------: | -------------: |
| 7 garden photos   | 0.785 → 0.899 | 0.797 → 0.860 | 0.805 → 0.811 | 0.731 → 0.731 |
| Synthetic streets | 0.690 → 0.944 | 0.746 → 0.899 | 0.793 → 0.837 | 0.770 → 0.773 |
| 9 real streets    | 0.849 → 0.941 | 0.864 → 0.907 | 0.866 → 0.860 | 0.788 → 0.788 |

The rule was then **fit with one scene type held out** (`python evaluate.py --fit-strength`); on every held-out type it beat the fixed default (scores 1.210 → 1.252, 1.215 → 1.301, 1.294 → 1.337; the per-image best, which needs the clear photo, is 1.283, 1.342, 1.418). The fit on all types gives slightly steeper constants (2.4 and 0.45) than the shipped 2.1 and 0.35, which were fit on an earlier, smaller run. On the **7 real fog photos**, which were not used for fitting, the gain is small and mixed: SSIM 0.706 → 0.718, PSNR 15.9 → 15.5 dB, edge F1 0.645 → 0.664.

### Are the defaults overfit?

Leave-one-scene-out on real fog (60 random settings; best on six scenes applied to the seventh): SSIM **0.784** and PSNR 17.9 dB, against 0.713 and 15.8 dB for the shipped default and 0.698 and 13.8 dB for the unprocessed photo. Settings tuned on other scenes beat the shipped default on unseen ones, so **the default is not optimal for real fog**; it was picked to keep the three modules active, and the **Fog removal only** and **Traffic monitoring** presets trade module strength for fidelity. Tuning used 7 scenes of one kind; do not read 0.784 as a benchmark.

### Blur estimation

`estimate_blur` measures the width of the strongest edges along **8 directions**, subtracts the width of a sharp photo in quadrature and reports the kind (Gaussian, or motion with its angle) and size. Gaussian and motion blur are told apart by how uneven the excess is across directions; the motion angle comes from a parabola fit to the directional peak. Fog scales an edge's step and gradient alike, so the estimate does not depend on fog density. `wiener_deblur` accepts any angle. Calibration was fit on the real photos only; the synthetic cities are the independent check.


| True blur                          | Real photos: estimate (px) |             kind right | Synthetic cities: estimate (px) | kind right |
| ---------------------------------- | -------------------------: | ---------------------: | ------------------------------: | ---------: |
| none (fog 0, 1.6, 3)               |                       0.00 |                  100 % |                            0.00 |      100 % |
| Gaussian 1 / 2 / 3 / 4 px          |  0.84 / 2.00 / 3.16 / 4.14 | 86 / 100 / 100 / 100 % |       1.06 / 1.78 / 2.47 / 3.48 |      100 % |
| Motion 1 / 2 / 3 / 4 px (7 angles) |  0.53 / 2.13 / 2.70 / 3.59 | 47 / 100 / 100 / 100 % |       1.05 / 2.13 / 2.52 / 2.98 |      100 % |

The motion angle is found to **about 3° from 2 px up** (mean error 2.9° real, 3.4° synthetic; 0.3° to 0.5° at 0° and 90°, up to about 6° at 25°/135°). The estimator found no blur in any of the 7 real hazy photos. Using the estimate in the pipeline keeps most of the benefit (fog + 45° motion blur, SSIM 0.629 with no deblur, 0.716 with the estimate, 0.770 with the true blur).

### Video

`dehaze_video.py` runs the pipeline on a video file (`python dehaze_video.py foggy.mp4 clear.mp4 --auto`). Per-frame processing flickers because the airlight, the transmission map, the stretch range and the colour balance move with sensor noise. `TemporalDehazer` smooths those estimates over time, smooths the transmission map only where nothing moved (so cars are not blended into the fog estimate), and averages the sensor noise on still pixels. On the synthetic traffic clip (flicker = mean frame-to-frame change of the output on static pixels, in grey levels):


| Processing                         |  Flicker | PSNR (dB) |  SSIM |
| ---------------------------------- | -------: | --------: | ----: |
| Foggy input                        |     1.54 |      12.4 | 0.747 |
| Every frame on its own             |     1.19 |      19.7 | 0.816 |
| **Temporal**                       | **0.42** |      19.7 | 0.810 |
| Temporal, strength from haze level |     0.34 |      16.9 | 0.828 |

That is about 65 % less flicker for the same fidelity, at about 0.09 s per 640 x 400 frame on one CPU. The app's **Video** card does this for an uploaded clip (limited frame count, H.264 output when `imageio-ffmpeg` is installed).

### What the numbers mean for the stated impact

- **Traffic monitoring.** In the test, fog hides almost a third of the vehicles and people from a detector (recall 0.71 on the untouched photo). The Traffic preset restores them to recall 0.90, where the full Balanced pipeline does not. The test uses simulated fog on real street photos and a small detector; it is a measured proxy for the impact, not a field trial.
- **Construction inspection.** Visible-edge gain, edge F1 and line recall (building and lane lines) rise strongly after processing, in every scene tested.

**Limitations**

- **No real hazy/clear street pairs.** The real-fog photos are gardens and parks; the real street photos are fogged by simulation with a rough depth guess (farther toward the top of the frame). Real fog on streets has not been measured. The 9 street photos are few and most are photos from one source.
- **No human study.** Metrics (PSNR, SSIM, Hautière visible-edge gain, edge F1, detector recall) stand in for human judgement and can disagree with it; contrast and edge counts also rise with over-sharpening. Judge the pictures as well.
- The adaptive strength rule was fit on simulated fog; on real fog its gain is small and mixed. The Balanced default is not optimal on real fog (see above).
- Blur estimation: blur under about 1 px reads as sharp (a 1 px motion blur is detected in under half of the real photos); large blur is under-read (about 0.5 to 1 px low at 4 px); defocus with a disc-shaped kernel and blur that varies across the frame are not handled. The sharp-photo baseline was measured on photos of about 800 to 1200 px; a resized or heavily compressed photo may read as slightly blurred or sharp.
- Detection is evaluation only (YOLOv5n, AGPL, not bundled and not part of the app); the app does not detect vehicles. The edge and line test uses one detector setting.
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
├── detect.py              # YOLOv5n vehicle/pedestrian detection for the evaluation (not used by the app)
├── get_street_photos.py   # Downloads the real street test photos
├── evaluate.py            # Ablation, street, detector, fog-density, blur, video experiments -> results/
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
    │   ├── urban_real_clear/  # real street photos (download with get_street_photos.py)
    │   └── urban_synthetic/   # 6 synthetic city streets: clear, hazy (β 1.6) and true depth map
    └── Preview/           # Demo recording (preview.gif + full .mp4 recording)
```

## Try it with the sample images

`assets/test images/` includes ready-to-use hazy photos. Drop any of them on the upload box:

- `city_smog/` – smoggy city streets and a foggy park
- `outdoor_fog_with_clear_reference/` – hazy scenes, each paired with a clear reference photo; `evaluate.py` measures them against their references
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

[ZannatulRaian](https://github.com/ZannatulRaian)

## Update — local contrast, adaptive fog removal, colour restoration

- **CLAHE stage (03b)**: contrast-limited adaptive histogram equalisation on the CIELAB L channel only (colours are not shifted), with clip-strength and tile-grid sliders. Runs after the histogram stretch. Off in "Course modules only".
- **Adaptive dark channel**: the airlight is the mean of the brightest fifth of the haze candidates (not one pixel, which can be a white car or a lamp), and the transmission floor t₀ is scaled to the measured haze. Toggle in stage 01.
- **Noise-aware sharpening**: the unsharp-mask detail is weighted by a local-activity map and the Immerkaer noise estimate, so edges are sharpened and flat noisy areas are not.
- **Colour restoration**: mid-tone white balance (removes the leftover blue/cyan cast) plus a saturation boost that is larger the more washed-out the image still is. Slider in stage 07.
- **Transmission view**: the estimated transmission map t as a colour map next to the original.
- **Measurements**: no-reference visibility rows (visible-edge gain e, gradient ratio r̄, newly clipped pixels) added to the table.
- **Fix**: an image enlarged with its hover button is no longer drawn under the Enhancement Pipeline card.
- **Real street photos**: `python get_street_photos.py` (already run; the 9 photos are in `assets/test images/urban_real_clear/`) and appear in the sample list with fog simulated on them.

On the 13 test scenes that have a clear reference, the new defaults score about the same PSNR/SSIM as before (17.4 dB / 0.74 vs 17.5 dB / 0.75): CLAHE and colour restoration trade a little pixel fidelity for local contrast and natural colour, so judge them by eye and by the visibility rows, not by PSNR alone.

**Bundled real photos** (for local testing; they keep their original licences): `assets/test images/urban_real_foggy/` has 7 real hazy/foggy scenes (street, town, skyline) that open as "already foggy"; `urban_real_clear/` has 10 clear street photos (with pedestrians, cars, a bus) that get simulated fog. Sources: github.com/matterport/Mask_RCNN, ultralytics/assets, and the sample images of Utkarsh-Deshmukh/Single-Image-Dehazing-Python and He-Zhang/image_dehaze.
