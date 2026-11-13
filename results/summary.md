# Evaluation results

7 real garden/park scenes with a clear reference, 10 real street photos fogged by simulation, 6 synthetic streets with true depth; photos scaled to at most 800px on the long edge. PSNR and SSIM are measured against the clear photo; the contrast gain is relative to the foggy input and edge pixels are the share of pixels with a gradient above 12. *Visible-edge gain e* and *gradient ratio r* (Hautiere et al.) compare a result with its foggy input and need no clear photo: e is the relative gain in visible edges, r the mean gradient ratio on them (above 1 = contrast gained). Each value is the mean over the scenes. The downstream test runs a standard Canny edge detector (fixed thresholds 50/150, which is what makes fog costly) and a probabilistic Hough line detector on every result: *Edge F1* compares its edges with the clear photo's (2 px tolerance); *auto thresholds* repeats that with Canny thresholds set from each image's median brightness, which compensates for low contrast and so shows the gain that is not just a brighter or higher-contrast picture; *line recall* is the share of the clear photo's lines found again; *vertical* lines are within 15 degrees of vertical (building edges) and *oblique* lines are 20-70 degrees from horizontal (lane and roof lines). More edges or lines is not better by itself: noise and over-sharpening add false ones, and the Hough line count also depends on how segments merge, so compare with F1 and recall.

## 1. Real fog (7 garden and park scenes)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 13.7 | 0.674 | 1.00 | 13.0 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 12.1 | 0.701 | 2.43 | 35.4 | +1.75 | 2.45 | 0.02 |
| Frequency-domain enhancement only | 13.8 | 0.739 | 1.08 | 25.7 | +0.68 | 1.62 | 0.05 |
| Adaptive smoothing + sharpening only | 13.7 | 0.601 | 1.01 | 8.3 | -0.44 | 0.95 | 0.05 |
| The three course modules together | 13.7 | 0.685 | 2.07 | 30.5 | +0.81 | 2.78 | 0.10 |
| Dark channel prior only | 16.9 | 0.774 | 1.23 | 23.6 | +0.85 | 1.68 | 0.10 |
| Full pipeline (Balanced default) | 15.2 | 0.693 | 2.09 | 41.0 | +1.57 | 3.96 | 0.31 |
| Full pipeline, strength set from the haze level | 14.6 | 0.719 | 2.00 | 34.5 | +1.35 | 3.13 | 0.37 |
| Traffic monitoring preset (fog removal + histogram stretching) | 14.9 | 0.733 | 1.95 | 38.6 | +1.95 | 2.71 | 0.02 |
| Traffic monitoring preset, strength set from the haze level | 14.8 | 0.736 | 1.97 | 35.2 | +1.85 | 2.46 | 0.19 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 0.4 | 0.155 | 0.078 | 6 | 3 | 2 | 0.039 | - |
| Histogram stretching only | 5.3 | 0.642 | 0.554 | 316 | 20 | 121 | 0.443 | 7/7 |
| Frequency-domain enhancement only | 2.1 | 0.473 | 0.245 | 118 | 7 | 51 | 0.185 | 7/7 |
| Adaptive smoothing + sharpening only | 0.9 | 0.291 | 0.140 | 22 | 5 | 7 | 0.070 | 7/7 |
| The three course modules together | 7.8 | 0.661 | 0.717 | 644 | 40 | 239 | 0.520 | 7/7 |
| Dark channel prior only | 2.0 | 0.462 | 0.689 | 65 | 6 | 26 | 0.166 | 7/7 |
| Full pipeline (Balanced default) | 11.9 | 0.597 | 0.724 | 1030 | 90 | 406 | 0.584 | 7/7 |
| Full pipeline, strength set from the haze level | 8.3 | 0.653 | 0.705 | 672 | 50 | 259 | 0.548 | 7/7 |
| Traffic monitoring preset (fog removal + histogram stretching) | 6.0 | 0.660 | 0.760 | 397 | 26 | 192 | 0.402 | 7/7 |
| Traffic monitoring preset, strength set from the haze level | 5.0 | 0.622 | 0.712 | 298 | 17 | 131 | 0.373 | 7/7 |

## 2. Real city streets, simulated fog

10 clear photos of real streets (`python get_street_photos.py`; one is `assets/test images/2467248.jpg`), fogged with the scattering model (beta 1.6) and a rough depth guess (farther toward the top of the frame). *Vehicles/people found* are the detections of a YOLOv5n detector (cars, buses, trucks, motorcycles, bicycles, pedestrians); recall and precision treat the detector's output on the clear photo as the reference (IoU 0.5). 

### 2. Real streets, fog beta 1.6

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.9 | 0.647 | 1.00 | 23.0 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 13.2 | 0.795 | 1.82 | 36.8 | +0.48 | 1.82 | 0.01 |
| Frequency-domain enhancement only | 11.1 | 0.724 | 1.08 | 35.5 | +0.40 | 1.61 | 0.03 |
| Adaptive smoothing + sharpening only | 11.0 | 0.614 | 1.06 | 18.8 | -0.25 | 1.15 | 0.03 |
| The three course modules together | 14.3 | 0.755 | 1.60 | 37.5 | +0.31 | 2.69 | 0.06 |
| Dark channel prior only | 21.1 | 0.917 | 1.49 | 39.2 | +0.48 | 2.02 | 0.06 |
| Full pipeline (Balanced default) | 17.9 | 0.810 | 1.78 | 47.2 | +0.63 | 3.89 | 0.17 |
| Full pipeline, strength set from the haze level | 17.3 | 0.817 | 1.73 | 44.6 | +0.53 | 3.39 | 0.22 |
| Traffic monitoring preset (fog removal + histogram stretching) | 21.0 | 0.922 | 1.77 | 43.4 | +0.63 | 2.40 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 19.2 | 0.898 | 1.77 | 42.3 | +0.58 | 2.27 | 0.10 |

Downstream test (Canny + Hough + vehicle/pedestrian detection, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input | Vehicles/people found | Detection recall vs clear | Detection precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 2.6 | 0.450 | 0.134 | 55 | 6 | 8 | 0.085 | - | 9.3 (clear photo: 11.3) | 0.731 | 0.914 |
| Histogram stretching only | 6.4 | 0.818 | 0.586 | 204 | 31 | 41 | 0.404 | 10/10 | 10.6 (clear photo: 11.3) | 0.848 | 0.918 |
| Frequency-domain enhancement only | 4.9 | 0.697 | 0.296 | 150 | 21 | 26 | 0.265 | 10/10 | 8.8 (clear photo: 11.3) | 0.664 | 0.889 |
| Adaptive smoothing + sharpening only | 4.1 | 0.634 | 0.284 | 116 | 16 | 17 | 0.198 | 10/10 | 8.7 (clear photo: 11.3) | 0.683 | 0.885 |
| The three course modules together | 9.2 | 0.846 | 0.757 | 388 | 57 | 96 | 0.527 | 10/10 | 8.3 (clear photo: 11.3) | 0.692 | 0.917 |
| Dark channel prior only | 7.7 | 0.895 | 0.876 | 262 | 44 | 72 | 0.480 | 10/10 | 10.9 (clear photo: 11.3) | 0.890 | 0.931 |
| Full pipeline (Balanced default) | 12.7 | 0.807 | 0.864 | 640 | 100 | 209 | 0.634 | 10/10 | 9.5 (clear photo: 11.3) | 0.761 | 0.925 |
| Full pipeline, strength set from the haze level | 11.7 | 0.848 | 0.836 | 565 | 93 | 191 | 0.650 | 10/10 | 9.5 (clear photo: 11.3) | 0.774 | 0.934 |
| Traffic monitoring preset (fog removal + histogram stretching) | 8.8 | 0.928 | 0.897 | 326 | 49 | 100 | 0.606 | 10/10 | 11.1 (clear photo: 11.3) | 0.886 | 0.951 |
| Traffic monitoring preset, strength set from the haze level | 8.5 | 0.914 | 0.849 | 308 | 50 | 86 | 0.545 | 10/10 | 11.3 (clear photo: 11.3) | 0.900 | 0.942 |

## 3. Simulated: fog

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.2 | 0.657 | 1.00 | 15.5 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 10.5 | 0.765 | 2.25 | 37.4 | +0.98 | 2.25 | 0.01 |
| Frequency-domain enhancement only | 10.2 | 0.732 | 1.07 | 29.4 | +0.59 | 1.63 | 0.05 |
| Adaptive smoothing + sharpening only | 10.2 | 0.568 | 1.01 | 9.9 | -0.44 | 0.95 | 0.05 |
| The three course modules together | 11.0 | 0.692 | 1.99 | 33.9 | +0.55 | 2.77 | 0.09 |
| Dark channel prior only | 21.0 | 0.943 | 1.33 | 37.5 | +0.95 | 2.24 | 0.10 |
| Full pipeline (Balanced default) | 16.3 | 0.743 | 1.96 | 52.4 | +1.25 | 4.87 | 0.29 |
| Full pipeline, strength set from the haze level | 15.7 | 0.759 | 1.92 | 50.0 | +1.18 | 4.45 | 0.36 |
| Traffic monitoring preset (fog removal + histogram stretching) | 18.7 | 0.895 | 1.86 | 48.4 | +1.30 | 3.13 | 0.02 |
| Traffic monitoring preset, strength set from the haze level | 17.5 | 0.891 | 1.87 | 47.0 | +1.27 | 3.00 | 0.17 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 0.5 | 0.165 | 0.011 | 13 | 1 | 3 | 0.035 | - |
| Histogram stretching only | 5.2 | 0.729 | 0.424 | 338 | 15 | 90 | 0.464 | 7/7 |
| Frequency-domain enhancement only | 2.3 | 0.476 | 0.091 | 152 | 6 | 38 | 0.200 | 7/7 |
| Adaptive smoothing + sharpening only | 1.1 | 0.326 | 0.052 | 48 | 3 | 7 | 0.105 | 7/7 |
| The three course modules together | 8.1 | 0.695 | 0.660 | 648 | 36 | 177 | 0.586 | 7/7 |
| Dark channel prior only | 4.4 | 0.852 | 0.827 | 218 | 21 | 86 | 0.511 | 7/7 |
| Full pipeline (Balanced default) | 14.9 | 0.548 | 0.710 | 1363 | 109 | 619 | 0.592 | 7/7 |
| Full pipeline, strength set from the haze level | 13.9 | 0.572 | 0.754 | 1231 | 120 | 563 | 0.573 | 7/7 |
| Traffic monitoring preset (fog removal + histogram stretching) | 8.4 | 0.780 | 0.818 | 621 | 53 | 255 | 0.660 | 7/7 |
| Traffic monitoring preset, strength set from the haze level | 7.9 | 0.799 | 0.822 | 559 | 47 | 217 | 0.666 | 7/7 |

## 3. Simulated: fog + Gaussian blur (2 px)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.1 | 0.411 | 1.00 | 0.1 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 10.2 | 0.485 | 2.44 | 2.6 | +8.75 | 2.45 | 0.01 |
| Frequency-domain enhancement only | 10.1 | 0.433 | 1.01 | 0.2 | +1.32 | 1.28 | 0.05 |
| Adaptive smoothing + sharpening only | 10.1 | 0.398 | 1.00 | 0.1 | -0.03 | 1.04 | 0.05 |
| The three course modules together | 10.8 | 0.492 | 2.23 | 4.7 | +9.96 | 2.65 | 0.09 |
| Dark channel prior only | 19.0 | 0.577 | 1.33 | 1.6 | +6.73 | 2.38 | 0.09 |
| Full pipeline (Balanced default) | 16.5 | 0.618 | 1.94 | 14.3 | +20.30 | 5.10 | 0.28 |
| Full pipeline, strength set from the haze level | 16.4 | 0.617 | 1.95 | 14.1 | +20.07 | 5.04 | 0.27 |
| Traffic monitoring preset (fog removal + histogram stretching) | 16.3 | 0.577 | 1.98 | 5.5 | +14.21 | 3.51 | 0.02 |
| Traffic monitoring preset, strength set from the haze level | 16.2 | 0.576 | 1.98 | 5.4 | +14.10 | 3.49 | 0.12 |
| Full pipeline + Wiener deblur (blur known) | 16.6 | 0.653 | 1.94 | 30.3 | +32.73 | 11.15 | 0.22 |
| Full pipeline + Wiener deblur (blur estimated) | 16.7 | 0.649 | 1.92 | 29.7 | +33.22 | 11.33 | 0.32 |
| Full pipeline, haze level and blur both estimated | 16.6 | 0.649 | 1.93 | 29.4 | +33.01 | 11.17 | 0.35 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 0.0 | 0.005 | 0.000 | 1 | 0 | 0 | 0.002 | - |
| Histogram stretching only | 0.7 | 0.245 | 0.078 | 13 | 3 | 2 | 0.061 | 7/7 |
| Frequency-domain enhancement only | 0.0 | 0.015 | 0.002 | 1 | 0 | 1 | 0.003 | 5/7 |
| Adaptive smoothing + sharpening only | 0.0 | 0.013 | 0.002 | 1 | 0 | 1 | 0.004 | 4/7 |
| The three course modules together | 1.4 | 0.403 | 0.171 | 27 | 5 | 4 | 0.093 | 7/7 |
| Dark channel prior only | 0.4 | 0.148 | 0.271 | 7 | 3 | 1 | 0.021 | 7/7 |
| Full pipeline (Balanced default) | 4.9 | 0.742 | 0.712 | 147 | 22 | 57 | 0.334 | 7/7 |
| Full pipeline, strength set from the haze level | 4.8 | 0.736 | 0.701 | 137 | 21 | 50 | 0.321 | 7/7 |
| Traffic monitoring preset (fog removal + histogram stretching) | 1.4 | 0.439 | 0.565 | 26 | 6 | 7 | 0.098 | 7/7 |
| Traffic monitoring preset, strength set from the haze level | 1.4 | 0.434 | 0.559 | 26 | 6 | 7 | 0.094 | 7/7 |
| Full pipeline + Wiener deblur (blur known) | 11.6 | 0.609 | 0.745 | 874 | 78 | 405 | 0.544 | 7/7 |
| Full pipeline + Wiener deblur (blur estimated) | 11.3 | 0.605 | 0.743 | 795 | 68 | 371 | 0.556 | 7/7 |
| Full pipeline, haze level and blur both estimated | 11.2 | 0.607 | 0.747 | 795 | 71 | 368 | 0.544 | 7/7 |

## 3. Simulated: fog + horizontal motion blur (2 px)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.1 | 0.478 | 1.00 | 3.0 | +0.00 | 1.00 | 0.01 |
| Histogram stretching only | 10.3 | 0.564 | 2.37 | 17.0 | +2.70 | 2.38 | 0.01 |
| Frequency-domain enhancement only | 10.1 | 0.519 | 1.03 | 9.1 | +1.14 | 1.53 | 0.04 |
| Adaptive smoothing + sharpening only | 10.1 | 0.444 | 1.00 | 1.9 | -0.45 | 0.96 | 0.05 |
| The three course modules together | 10.7 | 0.555 | 2.14 | 16.7 | +2.15 | 2.72 | 0.09 |
| Dark channel prior only | 19.7 | 0.684 | 1.32 | 14.4 | +2.44 | 2.30 | 0.10 |
| Full pipeline (Balanced default) | 16.4 | 0.640 | 1.95 | 33.2 | +4.29 | 5.06 | 0.29 |
| Full pipeline, strength set from the haze level | 16.1 | 0.642 | 1.94 | 32.2 | +4.20 | 4.88 | 0.30 |
| Traffic monitoring preset (fog removal + histogram stretching) | 17.1 | 0.668 | 1.93 | 25.7 | +3.94 | 3.36 | 0.02 |
| Traffic monitoring preset, strength set from the haze level | 16.9 | 0.668 | 1.93 | 25.2 | +3.90 | 3.30 | 0.14 |
| Full pipeline + Wiener deblur (blur known) | 16.5 | 0.725 | 1.97 | 52.7 | +6.17 | 7.85 | 0.23 |
| Full pipeline + Wiener deblur (blur estimated) | 16.5 | 0.717 | 1.97 | 53.8 | +6.23 | 7.97 | 0.35 |
| Full pipeline, haze level and blur both estimated | 16.3 | 0.723 | 1.96 | 52.7 | +6.16 | 7.70 | 0.41 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 0.1 | 0.026 | 0.000 | 1 | 0 | 1 | 0.005 | - |
| Histogram stretching only | 2.1 | 0.467 | 0.188 | 54 | 2 | 7 | 0.133 | 7/7 |
| Frequency-domain enhancement only | 0.4 | 0.125 | 0.009 | 7 | 1 | 1 | 0.030 | 7/7 |
| Adaptive smoothing + sharpening only | 0.2 | 0.064 | 0.006 | 3 | 0 | 1 | 0.007 | 7/7 |
| The three course modules together | 4.2 | 0.607 | 0.381 | 161 | 4 | 31 | 0.239 | 7/7 |
| Dark channel prior only | 1.2 | 0.370 | 0.503 | 23 | 3 | 2 | 0.081 | 7/7 |
| Full pipeline (Balanced default) | 9.9 | 0.631 | 0.753 | 526 | 37 | 218 | 0.464 | 7/7 |
| Full pipeline, strength set from the haze level | 9.4 | 0.639 | 0.757 | 497 | 30 | 207 | 0.472 | 7/7 |
| Traffic monitoring preset (fog removal + histogram stretching) | 3.8 | 0.678 | 0.743 | 103 | 9 | 27 | 0.238 | 7/7 |
| Traffic monitoring preset, strength set from the haze level | 3.7 | 0.672 | 0.712 | 98 | 9 | 28 | 0.236 | 7/7 |
| Full pipeline + Wiener deblur (blur known) | 13.8 | 0.562 | 0.715 | 1171 | 104 | 517 | 0.569 | 7/7 |
| Full pipeline + Wiener deblur (blur estimated) | 13.9 | 0.563 | 0.717 | 1176 | 86 | 526 | 0.559 | 7/7 |
| Full pipeline, haze level and blur both estimated | 13.5 | 0.572 | 0.732 | 1128 | 97 | 499 | 0.572 | 7/7 |

## 3. Simulated: fog + diagonal motion blur (45 deg, 2 px)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.1 | 0.445 | 1.00 | 1.1 | +0.00 | 1.00 | 0.01 |
| Histogram stretching only | 10.3 | 0.528 | 2.39 | 11.2 | +3.80 | 2.40 | 0.02 |
| Frequency-domain enhancement only | 10.1 | 0.478 | 1.02 | 4.4 | +1.30 | 1.46 | 0.05 |
| Adaptive smoothing + sharpening only | 10.1 | 0.424 | 1.00 | 0.8 | -0.37 | 0.98 | 0.05 |
| The three course modules together | 10.9 | 0.534 | 2.17 | 11.7 | +3.18 | 2.60 | 0.10 |
| Dark channel prior only | 19.4 | 0.635 | 1.32 | 8.2 | +3.35 | 2.30 | 0.09 |
| Full pipeline (Balanced default) | 16.6 | 0.635 | 1.93 | 26.4 | +6.71 | 4.92 | 0.28 |
| Full pipeline, strength set from the haze level | 16.3 | 0.633 | 1.93 | 25.6 | +6.54 | 4.76 | 0.30 |
| Traffic monitoring preset (fog removal + histogram stretching) | 16.8 | 0.627 | 1.93 | 18.1 | +5.82 | 3.37 | 0.02 |
| Traffic monitoring preset, strength set from the haze level | 16.5 | 0.625 | 1.94 | 17.7 | +5.72 | 3.31 | 0.16 |
| Full pipeline + Wiener deblur (blur known) | 16.5 | 0.714 | 1.95 | 51.5 | +10.64 | 9.02 | 0.23 |
| Full pipeline + Wiener deblur (blur estimated) | 16.5 | 0.666 | 1.95 | 50.9 | +10.66 | 8.81 | 0.34 |
| Full pipeline, haze level and blur both estimated | 16.3 | 0.671 | 1.94 | 49.8 | +10.54 | 8.52 | 0.39 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 0.1 | 0.026 | 0.005 | 2 | 1 | 1 | 0.005 | - |
| Histogram stretching only | 2.0 | 0.481 | 0.183 | 54 | 7 | 9 | 0.128 | 7/7 |
| Frequency-domain enhancement only | 0.3 | 0.097 | 0.011 | 3 | 1 | 1 | 0.013 | 7/7 |
| Adaptive smoothing + sharpening only | 0.1 | 0.047 | 0.009 | 2 | 1 | 1 | 0.006 | 7/7 |
| The three course modules together | 3.8 | 0.636 | 0.391 | 188 | 11 | 54 | 0.306 | 7/7 |
| Dark channel prior only | 1.1 | 0.356 | 0.485 | 17 | 5 | 4 | 0.057 | 7/7 |
| Full pipeline (Balanced default) | 8.9 | 0.680 | 0.753 | 588 | 47 | 285 | 0.546 | 7/7 |
| Full pipeline, strength set from the haze level | 8.6 | 0.687 | 0.747 | 556 | 47 | 264 | 0.505 | 7/7 |
| Traffic monitoring preset (fog removal + histogram stretching) | 3.4 | 0.674 | 0.715 | 101 | 16 | 43 | 0.248 | 7/7 |
| Traffic monitoring preset, strength set from the haze level | 3.3 | 0.664 | 0.684 | 97 | 16 | 38 | 0.223 | 7/7 |
| Full pipeline + Wiener deblur (blur known) | 13.7 | 0.568 | 0.719 | 1187 | 98 | 541 | 0.580 | 7/7 |
| Full pipeline + Wiener deblur (blur estimated) | 13.2 | 0.570 | 0.715 | 1154 | 105 | 537 | 0.580 | 7/7 |
| Full pipeline, haze level and blur both estimated | 12.9 | 0.578 | 0.729 | 1118 | 113 | 502 | 0.588 | 7/7 |

## 4. Synthetic city streets with true depth

6 synthetic streets (assets/test images/urban_synthetic, 640x400, different building layouts, cars, lane lines), fogged with t = exp(-beta * depth) from their true depth map.

### 4.1. Synthetic streets: fog (beta 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 12.2 | 0.754 | 1.00 | 6.9 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 12.5 | 0.773 | 1.56 | 7.1 | +0.03 | 1.56 | 0.01 |
| Frequency-domain enhancement only | 12.1 | 0.750 | 1.10 | 9.1 | +0.62 | 54.26 | 0.02 |
| Adaptive smoothing + sharpening only | 12.1 | 0.742 | 1.11 | 13.7 | +0.76 | 121.53 | 0.03 |
| The three course modules together | 12.2 | 0.696 | 1.31 | 24.7 | +1.42 | 749.24 | 0.05 |
| Dark channel prior only | 21.7 | 0.871 | 1.09 | 7.1 | +0.02 | 1.50 | 0.04 |
| Full pipeline (Balanced default) | 17.2 | 0.796 | 1.36 | 21.0 | +1.42 | 671.42 | 0.15 |
| Full pipeline, strength set from the haze level | 14.8 | 0.812 | 1.32 | 13.9 | +0.99 | 230.86 | 0.19 |
| Traffic monitoring preset (fog removal + histogram stretching) | 19.9 | 0.865 | 1.24 | 7.1 | +0.03 | 1.69 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 16.7 | 0.845 | 1.33 | 7.1 | +0.03 | 1.61 | 0.08 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 4.5 | 0.836 | 0.815 | 86 | 49 | 14 | 0.534 | - |
| Histogram stretching only | 5.7 | 0.934 | 0.951 | 120 | 48 | 16 | 0.770 | 6/6 |
| Frequency-domain enhancement only | 5.7 | 0.935 | 0.865 | 131 | 50 | 14 | 0.789 | 6/6 |
| Adaptive smoothing + sharpening only | 5.7 | 0.937 | 0.874 | 134 | 49 | 15 | 0.792 | 6/6 |
| The three course modules together | 5.8 | 0.929 | 0.950 | 125 | 32 | 16 | 0.792 | 6/6 |
| Dark channel prior only | 5.8 | 0.945 | 0.965 | 134 | 50 | 16 | 0.834 | 6/6 |
| Full pipeline (Balanced default) | 6.0 | 0.935 | 0.936 | 132 | 32 | 15 | 0.770 | 6/6 |
| Full pipeline, strength set from the haze level | 5.8 | 0.942 | 0.960 | 129 | 64 | 14 | 0.769 | 6/6 |
| Traffic monitoring preset (fog removal + histogram stretching) | 5.8 | 0.944 | 0.967 | 133 | 47 | 15 | 0.819 | 6/6 |
| Traffic monitoring preset, strength set from the haze level | 5.8 | 0.940 | 0.960 | 129 | 49 | 14 | 0.807 | 6/6 |

### 4.2. Synthetic streets: fog + horizontal motion blur (2 px) (beta 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 12.0 | 0.647 | 1.00 | 6.7 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 11.3 | 0.664 | 1.69 | 16.8 | +0.32 | 1.70 | 0.01 |
| Frequency-domain enhancement only | 12.0 | 0.659 | 1.04 | 16.0 | +0.40 | 2.67 | 0.02 |
| Adaptive smoothing + sharpening only | 12.0 | 0.662 | 1.04 | 12.9 | +0.21 | 3.81 | 0.02 |
| The three course modules together | 11.9 | 0.663 | 1.45 | 25.7 | +0.84 | 26.83 | 0.05 |
| Dark channel prior only | 20.6 | 0.776 | 1.01 | 16.5 | +0.31 | 1.44 | 0.04 |
| Full pipeline (Balanced default) | 16.4 | 0.733 | 1.33 | 26.7 | +0.90 | 26.23 | 0.13 |
| Full pipeline, strength set from the haze level | 13.7 | 0.719 | 1.37 | 21.4 | +0.60 | 7.18 | 0.16 |
| Traffic monitoring preset (fog removal + histogram stretching) | 19.1 | 0.759 | 1.21 | 18.6 | +0.37 | 1.73 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 15.9 | 0.743 | 1.35 | 18.0 | +0.36 | 1.65 | 0.08 |
| Full pipeline + Wiener deblur (blur known) | 17.3 | 0.728 | 1.40 | 26.9 | +1.26 | 84.02 | 0.10 |
| Full pipeline + Wiener deblur (blur estimated) | 17.2 | 0.732 | 1.40 | 27.2 | +1.22 | 80.46 | 0.15 |
| Full pipeline, haze level and blur both estimated | 14.1 | 0.748 | 1.36 | 19.5 | +0.79 | 38.55 | 0.17 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 4.6 | 0.778 | 0.616 | 116 | 8 | 14 | 0.523 | - |
| Histogram stretching only | 5.5 | 0.852 | 0.871 | 130 | 24 | 15 | 0.646 | 6/6 |
| Frequency-domain enhancement only | 4.4 | 0.813 | 0.782 | 106 | 33 | 15 | 0.600 | 6/6 |
| Adaptive smoothing + sharpening only | 4.8 | 0.804 | 0.770 | 107 | 14 | 14 | 0.570 | 6/6 |
| The three course modules together | 6.1 | 0.936 | 0.921 | 158 | 20 | 16 | 0.726 | 6/6 |
| Dark channel prior only | 5.8 | 0.897 | 0.947 | 150 | 16 | 16 | 0.712 | 6/6 |
| Full pipeline (Balanced default) | 6.5 | 0.940 | 0.948 | 167 | 22 | 16 | 0.749 | 6/6 |
| Full pipeline, strength set from the haze level | 6.6 | 0.931 | 0.926 | 182 | 20 | 16 | 0.734 | 6/6 |
| Traffic monitoring preset (fog removal + histogram stretching) | 6.4 | 0.931 | 0.947 | 167 | 20 | 16 | 0.737 | 6/6 |
| Traffic monitoring preset, strength set from the haze level | 6.3 | 0.922 | 0.937 | 164 | 18 | 15 | 0.714 | 6/6 |
| Full pipeline + Wiener deblur (blur known) | 6.6 | 0.892 | 0.863 | 155 | 33 | 24 | 0.753 | 6/6 |
| Full pipeline + Wiener deblur (blur estimated) | 6.5 | 0.897 | 0.872 | 154 | 32 | 23 | 0.753 | 6/6 |
| Full pipeline, haze level and blur both estimated | 5.9 | 0.938 | 0.954 | 134 | 26 | 16 | 0.765 | 6/6 |

### 4.3. Synthetic streets: fog + diagonal motion blur (45 deg, 2 px) (beta 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 12.0 | 0.633 | 1.00 | 17.1 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 11.4 | 0.648 | 1.68 | 22.1 | +0.12 | 1.68 | 0.01 |
| Frequency-domain enhancement only | 12.0 | 0.644 | 1.04 | 20.1 | +0.07 | 1.25 | 0.02 |
| Adaptive smoothing + sharpening only | 12.0 | 0.651 | 1.03 | 13.8 | -0.16 | 1.08 | 0.03 |
| The three course modules together | 12.8 | 0.665 | 1.56 | 26.4 | +0.30 | 4.12 | 0.05 |
| Dark channel prior only | 20.6 | 0.763 | 1.01 | 22.6 | +0.11 | 1.44 | 0.04 |
| Full pipeline (Balanced default) | 16.9 | 0.731 | 1.30 | 27.5 | +0.35 | 4.53 | 0.14 |
| Full pipeline, strength set from the haze level | 14.4 | 0.715 | 1.35 | 24.4 | +0.19 | 2.18 | 0.16 |
| Traffic monitoring preset (fog removal + histogram stretching) | 19.1 | 0.744 | 1.21 | 24.3 | +0.15 | 1.71 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 15.9 | 0.727 | 1.35 | 23.9 | +0.15 | 1.64 | 0.07 |
| Full pipeline + Wiener deblur (blur known) | 17.5 | 0.727 | 1.38 | 30.5 | +0.81 | 28.44 | 0.10 |
| Full pipeline + Wiener deblur (blur estimated) | 17.5 | 0.728 | 1.38 | 30.4 | +0.81 | 28.20 | 0.14 |
| Full pipeline, haze level and blur both estimated | 14.5 | 0.750 | 1.28 | 22.4 | +0.46 | 14.42 | 0.18 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 3.8 | 0.791 | 0.772 | 61 | 36 | 10 | 0.358 | - |
| Histogram stretching only | 5.5 | 0.941 | 0.939 | 119 | 39 | 14 | 0.732 | 6/6 |
| Frequency-domain enhancement only | 4.3 | 0.831 | 0.793 | 71 | 40 | 14 | 0.426 | 6/6 |
| Adaptive smoothing + sharpening only | 4.2 | 0.825 | 0.794 | 71 | 38 | 13 | 0.439 | 6/6 |
| The three course modules together | 5.9 | 0.938 | 0.948 | 116 | 37 | 15 | 0.715 | 6/6 |
| Dark channel prior only | 5.4 | 0.933 | 0.965 | 116 | 49 | 15 | 0.699 | 6/6 |
| Full pipeline (Balanced default) | 6.4 | 0.943 | 0.961 | 142 | 26 | 16 | 0.709 | 6/6 |
| Full pipeline, strength set from the haze level | 6.2 | 0.945 | 0.957 | 142 | 31 | 18 | 0.739 | 6/6 |
| Traffic monitoring preset (fog removal + histogram stretching) | 5.5 | 0.946 | 0.964 | 120 | 41 | 14 | 0.731 | 6/6 |
| Traffic monitoring preset, strength set from the haze level | 5.5 | 0.940 | 0.956 | 120 | 43 | 14 | 0.725 | 6/6 |
| Full pipeline + Wiener deblur (blur known) | 6.6 | 0.885 | 0.839 | 150 | 50 | 30 | 0.792 | 6/6 |
| Full pipeline + Wiener deblur (blur estimated) | 6.6 | 0.886 | 0.839 | 145 | 47 | 28 | 0.788 | 6/6 |
| Full pipeline, haze level and blur both estimated | 6.1 | 0.918 | 0.939 | 125 | 42 | 20 | 0.768 | 6/6 |

### 4.4. Synthetic streets: fog + Gaussian blur (2 px) (beta 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 11.9 | 0.609 | 1.00 | 8.5 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 11.2 | 0.636 | 1.71 | 18.9 | +0.32 | 1.71 | 0.01 |
| Frequency-domain enhancement only | 11.9 | 0.632 | 1.02 | 11.9 | +0.10 | 1.20 | 0.02 |
| Adaptive smoothing + sharpening only | 11.9 | 0.631 | 1.01 | 10.0 | -0.19 | 1.07 | 0.02 |
| The three course modules together | 12.3 | 0.708 | 1.62 | 17.7 | +0.11 | 1.90 | 0.04 |
| Dark channel prior only | 20.3 | 0.747 | 0.97 | 14.4 | +0.21 | 1.39 | 0.04 |
| Full pipeline (Balanced default) | 17.1 | 0.778 | 1.30 | 20.8 | +0.26 | 2.27 | 0.13 |
| Full pipeline, strength set from the haze level | 14.1 | 0.730 | 1.37 | 19.7 | +0.23 | 1.87 | 0.16 |
| Traffic monitoring preset (fog removal + histogram stretching) | 18.7 | 0.731 | 1.22 | 19.5 | +0.34 | 1.75 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 16.0 | 0.712 | 1.33 | 18.1 | +0.32 | 1.61 | 0.08 |
| Full pipeline + Wiener deblur (blur known) | 17.1 | 0.755 | 1.38 | 24.3 | +0.63 | 3.61 | 0.10 |
| Full pipeline + Wiener deblur (blur estimated) | 17.0 | 0.777 | 1.38 | 21.9 | +0.45 | 3.15 | 0.16 |
| Full pipeline, haze level and blur both estimated | 14.2 | 0.760 | 1.45 | 21.0 | +0.26 | 2.61 | 0.17 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 1.3 | 0.355 | 0.299 | 21 | 8 | 10 | 0.188 | - |
| Histogram stretching only | 4.4 | 0.834 | 0.827 | 83 | 48 | 12 | 0.490 | 6/6 |
| Frequency-domain enhancement only | 1.5 | 0.386 | 0.358 | 24 | 8 | 12 | 0.193 | 6/6 |
| Adaptive smoothing + sharpening only | 1.3 | 0.353 | 0.345 | 22 | 8 | 11 | 0.179 | 4/6 |
| The three course modules together | 4.8 | 0.843 | 0.895 | 91 | 45 | 14 | 0.537 | 6/6 |
| Dark channel prior only | 3.1 | 0.710 | 0.957 | 51 | 28 | 14 | 0.327 | 6/6 |
| Full pipeline (Balanced default) | 6.1 | 0.946 | 0.964 | 142 | 28 | 15 | 0.754 | 6/6 |
| Full pipeline, strength set from the haze level | 5.6 | 0.919 | 0.939 | 118 | 30 | 14 | 0.695 | 6/6 |
| Traffic monitoring preset (fog removal + histogram stretching) | 4.7 | 0.883 | 0.962 | 96 | 42 | 16 | 0.591 | 6/6 |
| Traffic monitoring preset, strength set from the haze level | 4.3 | 0.825 | 0.943 | 76 | 46 | 12 | 0.457 | 6/6 |
| Full pipeline + Wiener deblur (blur known) | 6.3 | 0.912 | 0.921 | 140 | 58 | 26 | 0.732 | 6/6 |
| Full pipeline + Wiener deblur (blur estimated) | 5.9 | 0.938 | 0.946 | 134 | 49 | 16 | 0.779 | 6/6 |
| Full pipeline, haze level and blur both estimated | 5.8 | 0.940 | 0.960 | 130 | 58 | 17 | 0.733 | 6/6 |

### Transmission estimate against the true depth

Mean over the synthetic set; the true transmission is exp(-beta * depth), clipped to the same minimum as the estimate.

| Beta | Transmission MAE | Correlation with true t | Airlight error (0-255) |
| ---: | ---: | ---: | ---: |
| 0.8 | 0.094 | 0.908 | 5.7 |
| 1.6 | 0.070 | 0.939 | 5.0 |
| 2.4 | 0.060 | 0.951 | 4.7 |
| 3.2 | 0.056 | 0.959 | 4.0 |

## 5. Fog density sweep: fixed default against strength set from the haze level

Fog is simulated at each density (garden photos and street photos with the rough depth guess, synthetic streets with their true depth). *Haze level* is `haze_level()` of the foggy image; *strength used* is the share of the default's strengths that `auto_strength_params` applies. The default's strengths are tuned for dense fog; on light haze they over-process.

| Scenes | Fog beta | Haze level | Strength used | Input SSIM | Fixed SSIM | Adaptive SSIM | Fixed PSNR | Adaptive PSNR | Fixed edge F1 | Adaptive edge F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 garden photos | 0.4 | 0.36 | 0.40 | 0.917 | 0.736 | 0.870 | 18.5 | 19.7 | 0.518 | 0.666 |
| 7 garden photos | 0.8 | 0.46 | 0.63 | 0.819 | 0.740 | 0.824 | 18.1 | 17.6 | 0.523 | 0.625 |
| 7 garden photos | 1.6 | 0.62 | 0.93 | 0.657 | 0.743 | 0.759 | 16.3 | 15.7 | 0.548 | 0.572 |
| 7 garden photos | 2.4 | 0.72 | 1.00 | 0.545 | 0.731 | 0.731 | 13.1 | 13.1 | 0.587 | 0.587 |
| 7 garden photos | 3.2 | 0.78 | 1.00 | 0.469 | 0.681 | 0.681 | 10.2 | 10.2 | 0.611 | 0.611 |
| synthetic streets | 0.4 | 0.26 | 0.25 | 0.948 | 0.825 | 0.929 | 20.7 | 21.0 | 0.929 | 0.959 |
| synthetic streets | 0.8 | 0.33 | 0.35 | 0.875 | 0.826 | 0.885 | 20.2 | 18.1 | 0.941 | 0.949 |
| synthetic streets | 1.6 | 0.44 | 0.57 | 0.754 | 0.796 | 0.812 | 17.2 | 14.8 | 0.935 | 0.942 |
| synthetic streets | 2.4 | 0.52 | 0.74 | 0.668 | 0.756 | 0.759 | 14.3 | 12.9 | 0.937 | 0.934 |
| synthetic streets | 3.2 | 0.58 | 0.87 | 0.605 | 0.718 | 0.714 | 12.3 | 11.6 | 0.929 | 0.927 |
| 10 real streets | 0.4 | 0.33 | 0.37 | 0.903 | 0.821 | 0.911 | 19.1 | 22.1 | 0.785 | 0.908 |
| 10 real streets | 0.8 | 0.43 | 0.56 | 0.807 | 0.818 | 0.872 | 19.1 | 19.8 | 0.791 | 0.887 |
| 10 real streets | 1.6 | 0.58 | 0.86 | 0.647 | 0.810 | 0.817 | 17.9 | 17.3 | 0.807 | 0.848 |
| 10 real streets | 2.4 | 0.68 | 0.99 | 0.533 | 0.793 | 0.792 | 15.7 | 15.6 | 0.824 | 0.828 |
| 10 real streets | 3.2 | 0.75 | 1.00 | 0.456 | 0.746 | 0.746 | 13.4 | 13.4 | 0.821 | 0.821 |

On the 7 **real** fog photos (section 1; real fog was not used for the fit), fixed against adaptive default: PSNR 15.24 / 14.56 dB, SSIM 0.693 / 0.719, edge F1 0.597 / 0.653.

### Fit of the rule and held-out check

The rule was fit on simulated fog of two scene types and tested on the third (python evaluate.py --fit-strength). Score is SSIM + PSNR/40; per-image best is the best of six fixed scalings chosen with knowledge of the clear photo.

Fit on all scene types: strength = clip(2.4 x haze - 0.45, 0.25, 1) (in dehaze_core: 2.1 and 0.35).

| Held-out scene type | Rule fit on the others | Score, fixed default | Score, adaptive | Score, per-image best | Light fog only (beta <= 0.8): fixed | adaptive |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 7 garden photos | 2.0 x haze - 0.30 | 1.210 | 1.252 | 1.283 | 1.276 | 1.411 |
| synthetic streets | 3.0 x haze - 0.95 | 1.215 | 1.301 | 1.342 | 1.198 | 1.513 |
| 9 real streets | 3.0 x haze - 0.60 | 1.294 | 1.337 | 1.418 | 1.314 | 1.446 |

## 6. What each module adds to the full pipeline

Each row removes one part of the Balanced pipeline; compare with the first row. SSIM and edge F1 are against the clear photo, e is the visible-edge gain over the foggy input; detection recall is the share of the clear street photos' vehicles and people that the YOLOv5n detector still finds.

| Pipeline | Real fog: SSIM | F1 | e | Synthetic streets: SSIM | F1 | e | Real streets: SSIM | F1 | e | detection recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full pipeline (Balanced default) | 0.693 | 0.597 | +1.57 | 0.796 | 0.935 | +1.42 | 0.810 | 0.807 | +0.63 | 0.761 |
| without histogram stretching | 0.754 | 0.642 | +1.13 | 0.797 | 0.935 | +1.42 | 0.809 | 0.822 | +0.60 | 0.781 |
| without frequency-domain enhancement | 0.732 | 0.630 | +1.27 | 0.838 | 0.937 | +1.15 | 0.853 | 0.854 | +0.50 | 0.832 |
| without adaptive smoothing + sharpening | 0.691 | 0.599 | +3.06 | 0.841 | 0.937 | +0.17 | 0.865 | 0.842 | +1.02 | 0.814 |
| without all three course modules | 0.801 | 0.696 | +1.70 | 0.886 | 0.938 | +0.04 | 0.885 | 0.918 | +0.70 | 0.866 |
| without the dark channel prior | 0.671 | 0.660 | +0.69 | 0.714 | 0.930 | +1.29 | 0.739 | 0.837 | +0.22 | 0.750 |
| without colour balance | 0.699 | 0.602 | +1.53 | 0.799 | 0.935 | +1.41 | 0.810 | 0.807 | +0.63 | 0.791 |
| without CLAHE | 0.714 | 0.621 | +1.30 | 0.804 | 0.934 | +1.32 | 0.832 | 0.843 | +0.55 | 0.761 |
| without colour restoration | 0.695 | 0.599 | +1.55 | 0.797 | 0.936 | +1.42 | 0.810 | 0.807 | +0.63 | 0.771 |
| without the adaptive dark channel | 0.694 | 0.599 | +1.54 | 0.796 | 0.935 | +1.42 | 0.814 | 0.813 | +0.61 | 0.761 |
| without the three course modules and CLAHE | 0.799 | 0.610 | +1.16 | 0.894 | 0.941 | +0.03 | 0.890 | 0.925 | +0.54 | 0.847 |
| without CLAHE, colour restoration and the adaptive dark channel | 0.716 | 0.624 | +1.25 | 0.805 | 0.935 | +1.32 | 0.836 | 0.848 | +0.53 | 0.751 |
| without frequency, sharpening, smoothing and colour balance (Traffic preset) | 0.733 | 0.660 | +1.95 | 0.865 | 0.944 | +0.03 | 0.922 | 0.928 | +0.63 | 0.886 |

## 7. Blur estimation

`estimate_blur` measures the width of the strongest edges along 8 directions and subtracts the width of a sharp photo. Each row blurs the clear photos by a known amount (fog added first, beta 1.6 unless the row says otherwise) and compares the estimate with it; motion blur rows pool 7 angles (0, 25, 45, 70, 90, 135, 160 degrees), length 3 x amount. The calibration factors were fit on the real photos, so the synthetic cities are an independent check.

| True blur | Fog beta | Real: estimate (px) | kind right | abs. error (px) | angle error (deg) | Synthetic: estimate (px) | kind right | abs. error (px) | angle error (deg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none (sharp) | 0 | 0.00 | 100% | 0.00 | - | 0.00 | 100% | 0.00 | - |
| none (sharp) | 1.6 | 0.00 | 100% | 0.00 | - | 0.00 | 100% | 0.00 | - |
| none (sharp) | 3 | 0.00 | 100% | 0.00 | - | 0.00 | 100% | 0.00 | - |
| Gaussian 1 px | 1.6 | 0.84 | 86% | 0.22 | - | 1.06 | 100% | 0.06 | - |
| Gaussian 2 px | 1.6 | 2.00 | 100% | 0.20 | - | 1.78 | 100% | 0.22 | - |
| Gaussian 3 px | 1.6 | 3.16 | 100% | 0.31 | - | 2.47 | 100% | 0.53 | - |
| Gaussian 4 px | 1.6 | 4.14 | 100% | 0.29 | - | 3.48 | 100% | 0.53 | - |
| Motion 1 px (7 angles) | 1.6 | 0.53 | 47% | 0.52 | 8.7 | 1.05 | 100% | 0.07 | 15.0 |
| Motion 2 px (7 angles) | 1.6 | 2.13 | 100% | 0.28 | 2.9 | 2.13 | 100% | 0.17 | 3.4 |
| Motion 3 px (7 angles) | 1.6 | 2.70 | 100% | 0.36 | 3.0 | 2.52 | 100% | 0.48 | 3.2 |
| Motion 4 px (7 angles) | 1.6 | 3.59 | 100% | 0.75 | 3.4 | 2.98 | 100% | 1.02 | 3.5 |
| Gaussian 2 px | 3 | 1.98 | 100% | 0.19 | - | 1.80 | 100% | 0.20 | - |
| Motion 2 px (7 angles) | 3 | 2.00 | 96% | 0.27 | 1.8 | 2.18 | 100% | 0.21 | 3.4 |

Motion angle, by true angle (blur 2 to 4 px, fog beta 1.6):

| True motion angle | Real: mean angle error (deg) | Real: estimate at 3 px | Synthetic: mean angle error (deg) | Synthetic: estimate at 3 px |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.3 | 2.91 | 0.5 | 2.42 |
| 25 | 3.1 | 2.73 | 6.7 | 2.34 |
| 45 | 5.5 | 2.51 | 4.3 | 2.61 |
| 70 | 4.9 | 2.64 | 2.2 | 2.82 |
| 90 | 0.4 | 2.89 | 0.2 | 2.65 |
| 135 | 5.8 | 2.48 | 5.0 | 2.48 |
| 160 | 1.8 | 2.73 | 4.5 | 2.34 |

On the 7 real hazy photos (true blur unknown) the estimator found: none 7 (largest estimate 0.00 px).

## 8. Are the defaults overfit? Leave-one-scene-out tuning (real fog)

60 random settings (strengths, dark channel omega, colour balance) are scored on the 7 real fog scenes at 400 px. For each scene, the best setting on the other six is applied to it. The shipped default was tuned on a subset of these photos, so its row is optimistic; the street and synthetic results above are data it never saw.

| Settings | SSIM | PSNR (dB) |
| --- | ---: | ---: |
| Shipped default (tuned on these photos) | 0.711 | 15.38 |
| Tuned on the other 6 scenes, applied to the held-out one (60 random candidates) | 0.739 | 15.20 |
| Best of the 60 candidates for each scene separately (upper bound) | 0.780 | 16.49 |
| Unprocessed | 0.698 | 13.81 |

## 9. Video: synthetic traffic clip

30 frames, 640x400, fixed camera, 3 cars driving across, fog (beta 1.6) from the true depth and sensor noise (sigma 2). *Flicker* is the mean frame-to-frame change of the output in pixels that did not change in the input (static background); the clip is processed by `dehaze_video.TemporalDehazer`.

| Processing | Flicker (grey levels) | PSNR (dB) | SSIM | Seconds/frame |
| --- | ---: | ---: | ---: | ---: |
| Foggy input | 1.54 | 12.4 | 0.747 | - |
| Every frame on its own | 1.45 | 16.9 | 0.787 | 0.17 |
| Temporal (smoothed estimates + denoising) | 0.72 | 17.0 | 0.793 | 0.18 |
| Every frame on its own, strength from haze level | 1.24 | 14.5 | 0.804 | 0.22 |
| Temporal, strength from haze level | 0.61 | 14.5 | 0.807 | 0.22 |
| Temporal, CLAHE off | 0.51 | 17.4 | 0.807 | 0.21 |
| Temporal, CLAHE off, strength from haze level | 0.38 | 14.7 | 0.817 | 0.23 |
