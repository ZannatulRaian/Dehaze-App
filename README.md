<div align="center">

# Urban Fog Cleaner ( Dehaze App )

**Upload a foggy city photo, tune a six-stage enhancement pipeline, check the result with measurements and histograms, and download it.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-dehazeapp.streamlit.app-C8102E?style=for-the-badge&logo=streamlit&logoColor=white)](https://dehazeapp.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-DIP-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

</div>

---

## 📷 Main View

![Urban Fog Cleaner preview](assets/Preview/preview.gif)

---

## Overview

**Urban Fog Cleaner** is a Streamlit web app that restores visibility in foggy and hazy urban photos (buildings, roads, vehicles). It was built as a Digital Image Processing course project.

Fog removal runs first using the **Dark Channel Prior**, then adaptive smoothing, adaptive sharpening, histogram stretching and a frequency-domain filter refine the result, and an optional gamma and saturation step finishes it. The original and dehazed images can be compared side by side, with a draggable divider, or one stage at a time, and the app measures the difference with contrast, entropy and gradient figures plus histogram charts.

**Where this helps:** traffic monitoring, construction inspection, and any scene where poor visibility hides structure.

## What is in the app

Everything is on the main page (no sidebar), under the pixel-art header strip.

**Preview card** (left)
- Upload a photo by dropping it on the box. **Change Image** starts over.
- **View** switch:
  - **Side by side** — original and dehazed.
  - **Slider** — one image with a divider you drag between original (left) and dehazed (right).
  - **Original** and **Dehazed** — one image at a time.
  - **Stages** — steps through the pipeline in the order it runs (Dark Channel, Smoothing, Sharpening, Histogram, Frequency, Tone) and shows the image before and after the stage you pick. It follows your sliders live.
- **Measurements**, below the images: contrast (standard deviation), entropy, mean gradient and edge-pixel percentage for the original and the dehazed image. Two histogram charts compare them: brightness (grey for original, red for dehazed) and R, G, B (dashed for original, solid for dehazed).
- The card is only as tall as its content and stays in view while you scroll the pipeline controls.

**Enhancement Pipeline card** (right)
- **00 Input** — tick *Photo is already foggy* for real fog. Leave it off to simulate fog on a clear photo with the fog density slider (I = J·t + A·(1 − t), depth assumed to grow toward the top of the frame).
- **01 Dark Channel Prior** — on/off, haze removed (ω), minimum transmission (t₀), patch size (0 = automatic).
- **02 Histogram Stretch** — strength.
- **03 Frequency-Domain Filter** — strength and cutoff radius.
- **04 Adaptive Smoothing** — strength.
- **05 Adaptive Sharpening** — strength.
- **06 Final Tone** — gamma and saturation (the defaults leave the image untouched).
- **Preset** (Balanced, Light haze, Dense fog, Real foggy photo, Maximum detail), **Auto-Enhance** (back to Balanced), **Reset** and **Download Result**.

The **Method** card at the bottom gives the fog model, the approach and the references.

## Live demo

👉 **https://dehazeapp.streamlit.app/**

## Features

- Drag-and-drop upload of foggy photos (JPG, PNG, WebP, GIF, BMP, TIFF, AVIF and more; HEIC with an optional extra)
- Five views: side by side, compare slider, original, dehazed and stage-by-stage
- Contrast, entropy, gradient and edge measurements with brightness and R, G, B histogram comparisons
- Six pipeline stages with live sliders, five presets, **Auto-Enhance** and **Reset**
- Optional fog simulation (atmospheric scattering model) for clear photos
- One-click **Download Result** as PNG
- Handles EXIF rotation, transparency, 16-bit/float images and CMYK/grayscale files
- Friendly error messages for unreadable files
- Pixel-art header strip of cars and clouds animated across the page

## How it works

The pipeline in `dehaze_app.py` runs in this order (the numbers in brackets are the stage numbers shown in the app):

| Step | Stage | What it does |
| ---- | ----- | ------------ |
| 1 | **Dark Channel Prior** [01] (on by default, can be switched off) | Estimates the airlight and a per-pixel transmission map, refined with a **guided filter** so it follows real edges, then recovers the haze-free scene `J = (I − A) / t + A` |
| 2 | **Adaptive Smoothing** [04] | Edge-preserving bilateral filter; smooths flat areas and the noise fog removal amplifies, keeps edges |
| 3 | **Adaptive Sharpening** [05] | Unsharp mask on luminance with the added detail capped, so strong edges don't overshoot into halos |
| 4 | **Histogram Stretch** [02] | Linear contrast stretch of luminance, clipping 0–2 % from each end |
| 5 | **Frequency-Domain Filter** [03] | FFT high-frequency emphasis with a smooth Gaussian high-pass (no ringing) and mirror padding (no border wrap-around) |
| 6 | **Final Tone** [06] | Optional gamma and colour saturation |

**Design notes**

- Stages 2–5 work on **luminance only** (the Y channel of YCrCb). Processing R, G and B separately shifts hues and paints coloured noise.
- Every stage returns its input untouched when its slider is at `0`, so **Reset** switches the four filters off and puts gamma and saturation back to neutral. The dark-channel step has its own on/off toggle.
- Images larger than **1200 px** on the long edge are downscaled before processing.
- The measurements need no reference photo. Contrast, entropy and gradient rise when structure becomes visible, but also when an image is over-sharpened, so read them together with the picture.

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

**Optional:** to open iPhone HEIC/HEIF photos, run `pip install pillow-heif`.

## Project structure

```
Dehaze-App/
├── dehaze_app.py          # Streamlit app + the full dehazing pipeline
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
- `outdoor_fog_with_clear_reference/` – hazy scenes, each paired with a clear reference photo you can open next to the result

## Tech stack

[Streamlit](https://streamlit.io/) · [OpenCV](https://opencv.org/) (headless) · [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) · [Pillow](https://python-pillow.org/)

## References

- K. He, J. Sun, X. Tang, *Single Image Haze Removal Using Dark Channel Prior*, CVPR 2009 (IEEE TPAMI 2011)
- K. He, J. Sun, X. Tang, *Guided Image Filtering*, ECCV 2010 (IEEE TPAMI 2013)
- S. G. Narasimhan, S. K. Nayar, *Vision and the Atmosphere*, IJCV 2002
- H. Koschmieder, *Theorie der horizontalen Sichtweite*, 1924
- R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018
- C. Tomasi, R. Manduchi, *Bilateral Filtering for Gray and Color Images*, ICCV 1998

## Author

**Zannatul Raian** · [@ZannatulRaian](https://github.com/ZannatulRaian)
