# Evaluation results

7 real garden/park scenes with a clear reference, 9 real street photos fogged by simulation, 6 synthetic streets with true depth; photos scaled to at most 800px on the long edge. PSNR and SSIM are measured against the clear photo; the contrast gain is relative to the foggy input and edge pixels are the share of pixels with a gradient above 12. *Visible-edge gain e* and *gradient ratio r* (Hautiere et al.) compare a result with its foggy input and need no clear photo: e is the relative gain in visible edges, r the mean gradient ratio on them (above 1 = contrast gained). Each value is the mean over the scenes. The downstream test runs a standard Canny edge detector (fixed thresholds 50/150, which is what makes fog costly) and a probabilistic Hough line detector on every result: *Edge F1* compares its edges with the clear photo's (2 px tolerance); *auto thresholds* repeats that with Canny thresholds set from each image's median brightness, which compensates for low contrast and so shows the gain that is not just a brighter or higher-contrast picture; *line recall* is the share of the clear photo's lines found again; *vertical* lines are within 15 degrees of vertical (building edges) and *oblique* lines are 20-70 degrees from horizontal (lane and roof lines). More edges or lines is not better by itself: noise and over-sharpening add false ones, and the Hough line count also depends on how segments merge, so compare with F1 and recall.

## 1. Real fog (7 garden and park scenes)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 13.7 | 0.674 | 1.00 | 13.0 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 12.1 | 0.701 | 2.43 | 35.4 | +1.75 | 2.45 | 0.01 |
| Frequency-domain enhancement only | 13.8 | 0.739 | 1.08 | 25.7 | +0.68 | 1.62 | 0.04 |
| Adaptive smoothing + sharpening only | 13.7 | 0.602 | 1.01 | 8.3 | -0.44 | 0.95 | 0.04 |
| The three course modules together | 13.7 | 0.684 | 2.07 | 30.8 | +0.89 | 2.76 | 0.07 |
| Dark channel prior only | 17.0 | 0.769 | 1.17 | 22.7 | +0.79 | 1.62 | 0.07 |
| Full pipeline (Balanced default) | 15.9 | 0.706 | 2.03 | 34.6 | +1.11 | 3.21 | 0.08 |
| Full pipeline, strength set from the haze level | 15.5 | 0.718 | 2.03 | 30.9 | +1.00 | 2.73 | 0.21 |
| Traffic monitoring preset (fog removal + histogram stretching) | 15.1 | 0.739 | 1.93 | 39.0 | +2.01 | 2.73 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 14.9 | 0.737 | 1.96 | 35.5 | +1.89 | 2.48 | 0.07 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 0.4 | 0.155 | 0.078 | 6 | 3 | 2 | 0.039 | - |
| Histogram stretching only | 5.3 | 0.642 | 0.554 | 316 | 20 | 121 | 0.443 | 7/7 |
| Frequency-domain enhancement only | 2.1 | 0.473 | 0.245 | 118 | 7 | 51 | 0.185 | 7/7 |
| Adaptive smoothing + sharpening only | 0.9 | 0.291 | 0.140 | 22 | 5 | 7 | 0.070 | 7/7 |
| The three course modules together | 7.8 | 0.661 | 0.717 | 642 | 41 | 230 | 0.513 | 7/7 |
| Dark channel prior only | 1.7 | 0.441 | 0.678 | 53 | 8 | 19 | 0.150 | 7/7 |
| Full pipeline (Balanced default) | 9.4 | 0.645 | 0.723 | 763 | 66 | 295 | 0.551 | 7/7 |
| Full pipeline, strength set from the haze level | 7.0 | 0.664 | 0.740 | 544 | 38 | 248 | 0.509 | 7/7 |
| Traffic monitoring preset (fog removal + histogram stretching) | 6.2 | 0.674 | 0.757 | 420 | 24 | 199 | 0.434 | 7/7 |
| Traffic monitoring preset, strength set from the haze level | 5.1 | 0.629 | 0.708 | 323 | 17 | 159 | 0.385 | 7/7 |

## 2. Real city streets, simulated fog

9 clear photos of real streets (`python get_street_photos.py`; one is `assets/test images/2467248.jpg`), fogged with the scattering model (beta 1.6) and a rough depth guess (farther toward the top of the frame). *Vehicles/people found* are the detections of a YOLOv5n detector (cars, buses, trucks, motorcycles, bicycles, pedestrians); recall and precision treat the detector's output on the clear photo as the reference (IoU 0.5). 

### 2. Real streets, fog beta 1.6

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.9 | 0.645 | 1.00 | 22.7 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 13.1 | 0.790 | 1.79 | 36.1 | +0.50 | 1.79 | 0.01 |
| Frequency-domain enhancement only | 11.1 | 0.723 | 1.08 | 35.1 | +0.42 | 1.60 | 0.02 |
| Adaptive smoothing + sharpening only | 11.0 | 0.613 | 1.06 | 18.7 | -0.25 | 1.15 | 0.02 |
| The three course modules together | 14.2 | 0.759 | 1.58 | 36.8 | +0.33 | 2.64 | 0.04 |
| Dark channel prior only | 19.2 | 0.905 | 1.30 | 37.3 | +0.46 | 1.89 | 0.04 |
| Full pipeline (Balanced default) | 18.8 | 0.866 | 1.62 | 42.5 | +0.48 | 3.20 | 0.05 |
| Full pipeline, strength set from the haze level | 18.2 | 0.860 | 1.62 | 40.7 | +0.43 | 2.89 | 0.12 |
| Traffic monitoring preset (fog removal + histogram stretching) | 20.0 | 0.918 | 1.57 | 42.3 | +0.65 | 2.29 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 18.6 | 0.892 | 1.59 | 41.0 | +0.60 | 2.17 | 0.04 |

Downstream test (Canny + Hough + vehicle/pedestrian detection, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input | Vehicles/people found | Detection recall vs clear | Detection precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 2.6 | 0.446 | 0.129 | 52 | 6 | 7 | 0.083 | - | 9.8 (clear photo: 11.9) | 0.714 | 0.927 |
| Histogram stretching only | 6.3 | 0.819 | 0.567 | 185 | 24 | 36 | 0.407 | 9/9 | 11.0 (clear photo: 11.9) | 0.831 | 0.930 |
| Frequency-domain enhancement only | 4.8 | 0.694 | 0.282 | 138 | 17 | 24 | 0.261 | 9/9 | 9.2 (clear photo: 11.9) | 0.639 | 0.899 |
| Adaptive smoothing + sharpening only | 4.1 | 0.631 | 0.278 | 110 | 14 | 18 | 0.194 | 9/9 | 9.1 (clear photo: 11.9) | 0.674 | 0.900 |
| The three course modules together | 9.0 | 0.854 | 0.756 | 350 | 40 | 77 | 0.527 | 9/9 | 8.8 (clear photo: 11.9) | 0.699 | 0.955 |
| Dark channel prior only | 7.3 | 0.874 | 0.865 | 222 | 35 | 53 | 0.430 | 9/9 | 11.2 (clear photo: 11.9) | 0.881 | 0.953 |
| Full pipeline (Balanced default) | 10.7 | 0.889 | 0.877 | 452 | 51 | 130 | 0.652 | 9/9 | 9.9 (clear photo: 11.9) | 0.725 | 0.950 |
| Full pipeline, strength set from the haze level | 10.1 | 0.900 | 0.841 | 407 | 47 | 124 | 0.607 | 9/9 | 10.0 (clear photo: 11.9) | 0.747 | 0.955 |
| Traffic monitoring preset (fog removal + histogram stretching) | 8.4 | 0.924 | 0.895 | 292 | 37 | 76 | 0.578 | 9/9 | 11.6 (clear photo: 11.9) | 0.895 | 0.975 |
| Traffic monitoring preset, strength set from the haze level | 8.1 | 0.911 | 0.836 | 267 | 38 | 73 | 0.539 | 9/9 | 11.4 (clear photo: 11.9) | 0.863 | 0.964 |

## 3. Simulated: fog

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.2 | 0.657 | 1.00 | 15.5 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 10.5 | 0.765 | 2.25 | 37.4 | +0.98 | 2.25 | 0.01 |
| Frequency-domain enhancement only | 10.2 | 0.732 | 1.07 | 29.4 | +0.59 | 1.63 | 0.03 |
| Adaptive smoothing + sharpening only | 10.2 | 0.569 | 1.01 | 9.9 | -0.44 | 0.95 | 0.03 |
| The three course modules together | 11.0 | 0.694 | 1.99 | 34.2 | +0.58 | 2.76 | 0.07 |
| Dark channel prior only | 21.5 | 0.943 | 1.25 | 36.0 | +0.91 | 2.13 | 0.07 |
| Full pipeline (Balanced default) | 18.0 | 0.805 | 1.91 | 47.0 | +1.04 | 3.99 | 0.08 |
| Full pipeline, strength set from the haze level | 17.2 | 0.811 | 1.90 | 45.1 | +0.99 | 3.73 | 0.20 |
| Traffic monitoring preset (fog removal + histogram stretching) | 19.0 | 0.895 | 1.82 | 48.6 | +1.31 | 3.14 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 17.7 | 0.891 | 1.83 | 47.1 | +1.27 | 2.99 | 0.07 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 0.5 | 0.165 | 0.011 | 13 | 1 | 3 | 0.035 | - |
| Histogram stretching only | 5.2 | 0.729 | 0.424 | 338 | 15 | 90 | 0.464 | 7/7 |
| Frequency-domain enhancement only | 2.3 | 0.476 | 0.091 | 152 | 6 | 38 | 0.200 | 7/7 |
| Adaptive smoothing + sharpening only | 1.1 | 0.326 | 0.052 | 48 | 3 | 7 | 0.105 | 7/7 |
| The three course modules together | 8.1 | 0.696 | 0.658 | 651 | 33 | 180 | 0.555 | 7/7 |
| Dark channel prior only | 3.9 | 0.822 | 0.809 | 184 | 18 | 72 | 0.457 | 7/7 |
| Full pipeline (Balanced default) | 12.3 | 0.625 | 0.739 | 1069 | 110 | 445 | 0.582 | 7/7 |
| Full pipeline, strength set from the haze level | 11.4 | 0.652 | 0.772 | 973 | 90 | 378 | 0.642 | 7/7 |
| Traffic monitoring preset (fog removal + histogram stretching) | 8.5 | 0.774 | 0.831 | 614 | 50 | 236 | 0.648 | 7/7 |
| Traffic monitoring preset, strength set from the haze level | 7.9 | 0.798 | 0.822 | 552 | 47 | 200 | 0.684 | 7/7 |

## 3. Simulated: fog + Gaussian blur (2 px)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.1 | 0.411 | 1.00 | 0.1 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 10.2 | 0.485 | 2.44 | 2.6 | +8.75 | 2.45 | 0.01 |
| Frequency-domain enhancement only | 10.1 | 0.433 | 1.01 | 0.2 | +1.32 | 1.28 | 0.04 |
| Adaptive smoothing + sharpening only | 10.1 | 0.399 | 1.00 | 0.1 | -0.03 | 1.04 | 0.04 |
| The three course modules together | 10.8 | 0.492 | 2.23 | 4.7 | +10.03 | 2.65 | 0.07 |
| Dark channel prior only | 19.1 | 0.573 | 1.24 | 1.3 | +6.13 | 2.23 | 0.07 |
| Full pipeline (Balanced default) | 16.4 | 0.578 | 1.95 | 8.1 | +13.56 | 3.78 | 0.08 |
| Full pipeline, strength set from the haze level | 16.3 | 0.577 | 1.96 | 8.0 | +13.45 | 3.75 | 0.16 |
| Traffic monitoring preset (fog removal + histogram stretching) | 16.4 | 0.580 | 1.94 | 5.6 | +14.75 | 3.50 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 16.3 | 0.579 | 1.95 | 5.6 | +14.63 | 3.48 | 0.07 |
| Full pipeline + Wiener deblur (blur known) | 17.2 | 0.670 | 1.91 | 21.5 | +25.38 | 7.94 | 0.13 |
| Full pipeline + Wiener deblur (blur estimated) | 17.2 | 0.665 | 1.91 | 20.7 | +25.79 | 8.01 | 0.19 |
| Full pipeline, haze level and blur both estimated | 17.1 | 0.665 | 1.91 | 20.5 | +25.66 | 7.92 | 0.25 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 0.0 | 0.005 | 0.000 | 1 | 0 | 0 | 0.002 | - |
| Histogram stretching only | 0.7 | 0.245 | 0.078 | 13 | 3 | 2 | 0.061 | 7/7 |
| Frequency-domain enhancement only | 0.0 | 0.015 | 0.002 | 1 | 0 | 1 | 0.003 | 5/7 |
| Adaptive smoothing + sharpening only | 0.0 | 0.013 | 0.002 | 1 | 0 | 1 | 0.004 | 4/7 |
| The three course modules together | 1.4 | 0.404 | 0.170 | 27 | 5 | 5 | 0.081 | 7/7 |
| Dark channel prior only | 0.3 | 0.108 | 0.233 | 6 | 3 | 1 | 0.017 | 7/7 |
| Full pipeline (Balanced default) | 2.6 | 0.632 | 0.650 | 57 | 12 | 18 | 0.193 | 7/7 |
| Full pipeline, strength set from the haze level | 2.5 | 0.627 | 0.638 | 55 | 13 | 18 | 0.192 | 7/7 |
| Traffic monitoring preset (fog removal + histogram stretching) | 1.4 | 0.441 | 0.548 | 25 | 7 | 6 | 0.099 | 7/7 |
| Traffic monitoring preset, strength set from the haze level | 1.4 | 0.437 | 0.538 | 25 | 7 | 6 | 0.095 | 7/7 |
| Full pipeline + Wiener deblur (blur known) | 8.2 | 0.739 | 0.791 | 524 | 51 | 207 | 0.600 | 7/7 |
| Full pipeline + Wiener deblur (blur estimated) | 7.9 | 0.714 | 0.790 | 449 | 39 | 178 | 0.558 | 7/7 |
| Full pipeline, haze level and blur both estimated | 7.8 | 0.716 | 0.791 | 441 | 37 | 174 | 0.565 | 7/7 |

## 3. Simulated: fog + horizontal motion blur (2 px)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.1 | 0.478 | 1.00 | 3.0 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 10.3 | 0.564 | 2.37 | 17.0 | +2.70 | 2.38 | 0.01 |
| Frequency-domain enhancement only | 10.1 | 0.519 | 1.03 | 9.1 | +1.14 | 1.53 | 0.03 |
| Adaptive smoothing + sharpening only | 10.1 | 0.444 | 1.00 | 1.9 | -0.45 | 0.96 | 0.03 |
| The three course modules together | 10.7 | 0.555 | 2.14 | 16.8 | +2.17 | 2.72 | 0.07 |
| Dark channel prior only | 19.9 | 0.681 | 1.24 | 13.3 | +2.31 | 2.18 | 0.07 |
| Full pipeline (Balanced default) | 17.2 | 0.655 | 1.92 | 25.7 | +3.26 | 3.92 | 0.08 |
| Full pipeline, strength set from the haze level | 17.0 | 0.655 | 1.93 | 25.1 | +3.20 | 3.81 | 0.17 |
| Traffic monitoring preset (fog removal + histogram stretching) | 17.3 | 0.671 | 1.89 | 26.0 | +3.96 | 3.37 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 17.0 | 0.669 | 1.90 | 25.4 | +3.91 | 3.30 | 0.07 |
| Full pipeline + Wiener deblur (blur known) | 17.9 | 0.782 | 1.93 | 44.4 | +5.19 | 6.21 | 0.11 |
| Full pipeline + Wiener deblur (blur estimated) | 17.8 | 0.774 | 1.93 | 45.7 | +5.29 | 6.35 | 0.20 |
| Full pipeline, haze level and blur both estimated | 17.5 | 0.777 | 1.93 | 44.9 | +5.23 | 6.19 | 0.27 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 0.1 | 0.026 | 0.000 | 1 | 0 | 1 | 0.005 | - |
| Histogram stretching only | 2.1 | 0.467 | 0.188 | 54 | 2 | 7 | 0.133 | 7/7 |
| Frequency-domain enhancement only | 0.4 | 0.125 | 0.009 | 7 | 1 | 1 | 0.030 | 7/7 |
| Adaptive smoothing + sharpening only | 0.2 | 0.064 | 0.006 | 3 | 0 | 1 | 0.007 | 7/7 |
| The three course modules together | 4.2 | 0.603 | 0.381 | 156 | 4 | 31 | 0.235 | 7/7 |
| Dark channel prior only | 1.0 | 0.321 | 0.473 | 17 | 2 | 2 | 0.061 | 7/7 |
| Full pipeline (Balanced default) | 6.7 | 0.703 | 0.766 | 299 | 15 | 111 | 0.422 | 7/7 |
| Full pipeline, strength set from the haze level | 6.4 | 0.710 | 0.769 | 277 | 15 | 102 | 0.401 | 7/7 |
| Traffic monitoring preset (fog removal + histogram stretching) | 4.0 | 0.684 | 0.745 | 112 | 8 | 32 | 0.257 | 7/7 |
| Traffic monitoring preset, strength set from the haze level | 3.8 | 0.676 | 0.710 | 104 | 6 | 30 | 0.245 | 7/7 |
| Full pipeline + Wiener deblur (blur known) | 10.8 | 0.654 | 0.750 | 842 | 76 | 338 | 0.606 | 7/7 |
| Full pipeline + Wiener deblur (blur estimated) | 10.9 | 0.653 | 0.751 | 852 | 74 | 333 | 0.622 | 7/7 |
| Full pipeline, haze level and blur both estimated | 10.4 | 0.667 | 0.763 | 817 | 71 | 316 | 0.635 | 7/7 |

## 3. Simulated: fog + diagonal motion blur (45 deg, 2 px)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.1 | 0.445 | 1.00 | 1.1 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 10.3 | 0.528 | 2.39 | 11.2 | +3.80 | 2.40 | 0.01 |
| Frequency-domain enhancement only | 10.1 | 0.478 | 1.02 | 4.4 | +1.30 | 1.46 | 0.04 |
| Adaptive smoothing + sharpening only | 10.1 | 0.425 | 1.00 | 0.8 | -0.37 | 0.98 | 0.03 |
| The three course modules together | 10.9 | 0.534 | 2.17 | 11.8 | +3.22 | 2.61 | 0.07 |
| Dark channel prior only | 19.8 | 0.632 | 1.22 | 7.3 | +3.12 | 2.16 | 0.07 |
| Full pipeline (Balanced default) | 17.0 | 0.629 | 1.90 | 18.5 | +4.75 | 3.69 | 0.08 |
| Full pipeline, strength set from the haze level | 16.8 | 0.626 | 1.92 | 18.0 | +4.65 | 3.61 | 0.19 |
| Traffic monitoring preset (fog removal + histogram stretching) | 17.0 | 0.630 | 1.88 | 18.7 | +5.90 | 3.40 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 16.7 | 0.628 | 1.90 | 18.2 | +5.79 | 3.33 | 0.07 |
| Full pipeline + Wiener deblur (blur known) | 17.7 | 0.770 | 1.91 | 42.5 | +9.00 | 7.10 | 0.12 |
| Full pipeline + Wiener deblur (blur estimated) | 17.5 | 0.716 | 1.91 | 41.9 | +9.03 | 6.91 | 0.22 |
| Full pipeline, haze level and blur both estimated | 17.3 | 0.717 | 1.92 | 41.0 | +8.93 | 6.74 | 0.27 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 0.1 | 0.026 | 0.005 | 2 | 1 | 1 | 0.005 | - |
| Histogram stretching only | 2.0 | 0.481 | 0.183 | 54 | 7 | 9 | 0.128 | 7/7 |
| Frequency-domain enhancement only | 0.3 | 0.097 | 0.011 | 3 | 1 | 1 | 0.013 | 7/7 |
| Adaptive smoothing + sharpening only | 0.1 | 0.047 | 0.009 | 2 | 1 | 1 | 0.006 | 7/7 |
| The three course modules together | 3.8 | 0.636 | 0.391 | 180 | 11 | 51 | 0.291 | 7/7 |
| Dark channel prior only | 0.9 | 0.314 | 0.454 | 15 | 4 | 4 | 0.053 | 7/7 |
| Full pipeline (Balanced default) | 5.9 | 0.750 | 0.760 | 312 | 28 | 131 | 0.442 | 7/7 |
| Full pipeline, strength set from the haze level | 5.7 | 0.755 | 0.744 | 291 | 28 | 121 | 0.431 | 7/7 |
| Traffic monitoring preset (fog removal + histogram stretching) | 3.7 | 0.679 | 0.713 | 121 | 16 | 55 | 0.250 | 7/7 |
| Traffic monitoring preset, strength set from the haze level | 3.6 | 0.676 | 0.680 | 107 | 14 | 46 | 0.246 | 7/7 |
| Full pipeline + Wiener deblur (blur known) | 10.6 | 0.664 | 0.759 | 843 | 81 | 328 | 0.586 | 7/7 |
| Full pipeline + Wiener deblur (blur estimated) | 10.0 | 0.665 | 0.755 | 789 | 82 | 299 | 0.617 | 7/7 |
| Full pipeline, haze level and blur both estimated | 9.6 | 0.676 | 0.765 | 753 | 77 | 289 | 0.627 | 7/7 |

## 4. Synthetic city streets with true depth

6 synthetic streets (assets/test images/urban_synthetic, 640x400, different building layouts, cars, lane lines), fogged with t = exp(-beta * depth) from their true depth map.

### 4.1. Synthetic streets: fog (beta 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 12.2 | 0.754 | 1.00 | 6.9 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 12.5 | 0.773 | 1.56 | 7.1 | +0.03 | 1.56 | 0.01 |
| Frequency-domain enhancement only | 12.1 | 0.750 | 1.10 | 9.1 | +0.62 | 54.26 | 0.02 |
| Adaptive smoothing + sharpening only | 12.1 | 0.742 | 1.11 | 13.7 | +0.76 | 121.53 | 0.02 |
| The three course modules together | 12.2 | 0.696 | 1.31 | 24.7 | +1.42 | 749.20 | 0.04 |
| Dark channel prior only | 21.8 | 0.871 | 1.08 | 7.1 | +0.02 | 1.50 | 0.03 |
| Full pipeline (Balanced default) | 19.4 | 0.793 | 1.29 | 16.4 | +1.16 | 366.49 | 0.05 |
| Full pipeline, strength set from the haze level | 17.4 | 0.837 | 1.31 | 10.6 | +0.84 | 139.75 | 0.10 |
| Traffic monitoring preset (fog removal + histogram stretching) | 20.2 | 0.866 | 1.22 | 7.1 | +0.03 | 1.68 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 16.8 | 0.847 | 1.32 | 7.1 | +0.03 | 1.60 | 0.03 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 4.5 | 0.836 | 0.815 | 86 | 49 | 14 | 0.534 | - |
| Histogram stretching only | 5.7 | 0.934 | 0.951 | 120 | 48 | 16 | 0.770 | 6/6 |
| Frequency-domain enhancement only | 5.7 | 0.935 | 0.865 | 131 | 50 | 14 | 0.789 | 6/6 |
| Adaptive smoothing + sharpening only | 5.7 | 0.937 | 0.874 | 134 | 49 | 15 | 0.792 | 6/6 |
| The three course modules together | 5.8 | 0.930 | 0.950 | 123 | 30 | 16 | 0.771 | 6/6 |
| Dark channel prior only | 5.8 | 0.943 | 0.965 | 134 | 49 | 15 | 0.803 | 6/6 |
| Full pipeline (Balanced default) | 5.9 | 0.938 | 0.939 | 126 | 24 | 14 | 0.771 | 6/6 |
| Full pipeline, strength set from the haze level | 5.9 | 0.941 | 0.962 | 135 | 47 | 14 | 0.818 | 6/6 |
| Traffic monitoring preset (fog removal + histogram stretching) | 5.8 | 0.944 | 0.965 | 130 | 48 | 14 | 0.827 | 6/6 |
| Traffic monitoring preset, strength set from the haze level | 5.8 | 0.937 | 0.960 | 126 | 50 | 14 | 0.810 | 6/6 |

### 4.2. Synthetic streets: fog + horizontal motion blur (2 px) (beta 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 12.0 | 0.647 | 1.00 | 6.7 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 11.3 | 0.664 | 1.69 | 16.8 | +0.32 | 1.70 | 0.01 |
| Frequency-domain enhancement only | 12.0 | 0.659 | 1.04 | 16.0 | +0.40 | 2.67 | 0.02 |
| Adaptive smoothing + sharpening only | 12.0 | 0.662 | 1.04 | 12.9 | +0.21 | 3.81 | 0.02 |
| The three course modules together | 11.9 | 0.663 | 1.45 | 25.7 | +0.84 | 26.83 | 0.04 |
| Dark channel prior only | 20.7 | 0.776 | 1.00 | 16.5 | +0.31 | 1.43 | 0.03 |
| Full pipeline (Balanced default) | 18.8 | 0.752 | 1.29 | 22.2 | +0.72 | 13.87 | 0.04 |
| Full pipeline, strength set from the haze level | 15.8 | 0.744 | 1.38 | 19.5 | +0.55 | 5.67 | 0.10 |
| Traffic monitoring preset (fog removal + histogram stretching) | 19.2 | 0.759 | 1.20 | 18.5 | +0.37 | 1.72 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 15.9 | 0.743 | 1.34 | 18.0 | +0.36 | 1.64 | 0.03 |
| Full pipeline + Wiener deblur (blur known) | 19.4 | 0.756 | 1.37 | 22.3 | +0.99 | 49.32 | 0.05 |
| Full pipeline + Wiener deblur (blur estimated) | 19.3 | 0.759 | 1.37 | 22.3 | +0.97 | 47.99 | 0.09 |
| Full pipeline, haze level and blur both estimated | 16.3 | 0.801 | 1.39 | 17.6 | +0.64 | 26.23 | 0.12 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 4.6 | 0.778 | 0.616 | 116 | 8 | 14 | 0.523 | - |
| Histogram stretching only | 5.5 | 0.852 | 0.871 | 130 | 24 | 15 | 0.646 | 6/6 |
| Frequency-domain enhancement only | 4.4 | 0.813 | 0.782 | 106 | 33 | 15 | 0.600 | 6/6 |
| Adaptive smoothing + sharpening only | 4.8 | 0.804 | 0.770 | 107 | 14 | 14 | 0.570 | 6/6 |
| The three course modules together | 6.1 | 0.936 | 0.921 | 158 | 20 | 16 | 0.726 | 6/6 |
| Dark channel prior only | 5.8 | 0.897 | 0.946 | 154 | 17 | 17 | 0.705 | 6/6 |
| Full pipeline (Balanced default) | 6.2 | 0.945 | 0.948 | 154 | 22 | 16 | 0.739 | 6/6 |
| Full pipeline, strength set from the haze level | 6.1 | 0.936 | 0.944 | 154 | 22 | 16 | 0.741 | 6/6 |
| Traffic monitoring preset (fog removal + histogram stretching) | 6.4 | 0.929 | 0.946 | 168 | 20 | 15 | 0.739 | 6/6 |
| Traffic monitoring preset, strength set from the haze level | 6.3 | 0.923 | 0.936 | 168 | 18 | 17 | 0.709 | 6/6 |
| Full pipeline + Wiener deblur (blur known) | 6.2 | 0.920 | 0.883 | 136 | 28 | 16 | 0.751 | 6/6 |
| Full pipeline + Wiener deblur (blur estimated) | 6.2 | 0.922 | 0.889 | 128 | 28 | 17 | 0.737 | 6/6 |
| Full pipeline, haze level and blur both estimated | 5.9 | 0.938 | 0.953 | 134 | 28 | 15 | 0.755 | 6/6 |

### 4.3. Synthetic streets: fog + diagonal motion blur (45 deg, 2 px) (beta 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 12.0 | 0.633 | 1.00 | 17.1 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 11.4 | 0.648 | 1.68 | 22.1 | +0.12 | 1.68 | 0.00 |
| Frequency-domain enhancement only | 12.0 | 0.644 | 1.04 | 20.1 | +0.07 | 1.25 | 0.02 |
| Adaptive smoothing + sharpening only | 12.0 | 0.651 | 1.03 | 13.8 | -0.16 | 1.08 | 0.02 |
| The three course modules together | 12.8 | 0.664 | 1.56 | 26.4 | +0.30 | 4.12 | 0.03 |
| Dark channel prior only | 20.6 | 0.763 | 1.00 | 22.5 | +0.11 | 1.44 | 0.03 |
| Full pipeline (Balanced default) | 18.9 | 0.735 | 1.27 | 24.3 | +0.19 | 2.50 | 0.04 |
| Full pipeline, strength set from the haze level | 16.4 | 0.725 | 1.34 | 23.7 | +0.13 | 1.75 | 0.09 |
| Traffic monitoring preset (fog removal + histogram stretching) | 19.2 | 0.745 | 1.20 | 24.2 | +0.15 | 1.70 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 15.9 | 0.727 | 1.34 | 23.9 | +0.15 | 1.63 | 0.03 |
| Full pipeline + Wiener deblur (blur known) | 19.6 | 0.756 | 1.35 | 22.4 | +0.55 | 15.09 | 0.06 |
| Full pipeline + Wiener deblur (blur estimated) | 19.6 | 0.757 | 1.35 | 22.4 | +0.55 | 14.92 | 0.09 |
| Full pipeline, haze level and blur both estimated | 17.3 | 0.815 | 1.30 | 18.5 | +0.28 | 10.74 | 0.11 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 3.8 | 0.791 | 0.772 | 61 | 36 | 10 | 0.358 | - |
| Histogram stretching only | 5.5 | 0.941 | 0.939 | 119 | 39 | 14 | 0.732 | 6/6 |
| Frequency-domain enhancement only | 4.3 | 0.831 | 0.793 | 71 | 40 | 14 | 0.426 | 6/6 |
| Adaptive smoothing + sharpening only | 4.2 | 0.825 | 0.794 | 71 | 38 | 13 | 0.439 | 6/6 |
| The three course modules together | 5.9 | 0.938 | 0.948 | 116 | 37 | 15 | 0.715 | 6/6 |
| Dark channel prior only | 5.3 | 0.932 | 0.965 | 117 | 50 | 14 | 0.713 | 6/6 |
| Full pipeline (Balanced default) | 5.9 | 0.949 | 0.965 | 122 | 41 | 14 | 0.728 | 6/6 |
| Full pipeline, strength set from the haze level | 5.8 | 0.944 | 0.959 | 125 | 48 | 15 | 0.748 | 6/6 |
| Traffic monitoring preset (fog removal + histogram stretching) | 5.4 | 0.943 | 0.964 | 120 | 42 | 14 | 0.741 | 6/6 |
| Traffic monitoring preset, strength set from the haze level | 5.5 | 0.939 | 0.956 | 120 | 43 | 16 | 0.713 | 6/6 |
| Full pipeline + Wiener deblur (blur known) | 6.3 | 0.907 | 0.853 | 135 | 54 | 20 | 0.790 | 6/6 |
| Full pipeline + Wiener deblur (blur estimated) | 6.3 | 0.907 | 0.854 | 137 | 51 | 21 | 0.794 | 6/6 |
| Full pipeline, haze level and blur both estimated | 5.9 | 0.929 | 0.936 | 123 | 48 | 15 | 0.770 | 6/6 |

### 4.4. Synthetic streets: fog + Gaussian blur (2 px) (beta 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 11.9 | 0.609 | 1.00 | 8.5 | +0.00 | 1.00 | 0.00 |
| Histogram stretching only | 11.2 | 0.636 | 1.71 | 18.9 | +0.32 | 1.71 | 0.01 |
| Frequency-domain enhancement only | 11.9 | 0.632 | 1.02 | 11.9 | +0.10 | 1.20 | 0.02 |
| Adaptive smoothing + sharpening only | 11.9 | 0.631 | 1.01 | 10.0 | -0.19 | 1.07 | 0.02 |
| The three course modules together | 12.3 | 0.707 | 1.62 | 17.7 | +0.11 | 1.90 | 0.04 |
| Dark channel prior only | 20.3 | 0.747 | 0.96 | 14.3 | +0.21 | 1.38 | 0.03 |
| Full pipeline (Balanced default) | 19.5 | 0.772 | 1.22 | 18.1 | +0.11 | 1.93 | 0.04 |
| Full pipeline, strength set from the haze level | 16.1 | 0.739 | 1.36 | 18.4 | +0.14 | 1.75 | 0.11 |
| Traffic monitoring preset (fog removal + histogram stretching) | 18.6 | 0.732 | 1.22 | 19.7 | +0.35 | 1.76 | 0.01 |
| Traffic monitoring preset, strength set from the haze level | 16.0 | 0.712 | 1.32 | 18.1 | +0.32 | 1.61 | 0.03 |
| Full pipeline + Wiener deblur (blur known) | 19.2 | 0.777 | 1.35 | 21.2 | +0.35 | 3.09 | 0.06 |
| Full pipeline + Wiener deblur (blur estimated) | 19.1 | 0.786 | 1.36 | 19.7 | +0.21 | 2.78 | 0.10 |
| Full pipeline, haze level and blur both estimated | 16.1 | 0.772 | 1.46 | 20.0 | +0.16 | 2.61 | 0.12 |

Downstream test (Canny + Hough, means per image):

| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) | Oblique (lane) | Line recall vs clear | F1 above input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 1.3 | 0.355 | 0.299 | 21 | 8 | 10 | 0.188 | - |
| Histogram stretching only | 4.4 | 0.834 | 0.827 | 83 | 48 | 12 | 0.490 | 6/6 |
| Frequency-domain enhancement only | 1.5 | 0.386 | 0.358 | 24 | 8 | 12 | 0.193 | 6/6 |
| Adaptive smoothing + sharpening only | 1.3 | 0.353 | 0.345 | 22 | 8 | 11 | 0.179 | 4/6 |
| The three course modules together | 4.8 | 0.843 | 0.895 | 90 | 45 | 14 | 0.536 | 6/6 |
| Dark channel prior only | 3.1 | 0.709 | 0.956 | 49 | 29 | 12 | 0.323 | 6/6 |
| Full pipeline (Balanced default) | 5.6 | 0.937 | 0.964 | 127 | 32 | 16 | 0.762 | 6/6 |
| Full pipeline, strength set from the haze level | 5.1 | 0.898 | 0.945 | 100 | 39 | 14 | 0.640 | 6/6 |
| Traffic monitoring preset (fog removal + histogram stretching) | 4.7 | 0.885 | 0.962 | 92 | 41 | 14 | 0.594 | 6/6 |
| Traffic monitoring preset, strength set from the haze level | 4.3 | 0.824 | 0.943 | 78 | 47 | 12 | 0.442 | 6/6 |
| Full pipeline + Wiener deblur (blur known) | 5.9 | 0.937 | 0.933 | 128 | 23 | 19 | 0.715 | 6/6 |
| Full pipeline + Wiener deblur (blur estimated) | 5.8 | 0.942 | 0.962 | 119 | 38 | 15 | 0.728 | 6/6 |
| Full pipeline, haze level and blur both estimated | 5.7 | 0.941 | 0.961 | 120 | 51 | 15 | 0.716 | 6/6 |

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
| 7 garden photos | 0.4 | 0.36 | 0.40 | 0.917 | 0.785 | 0.899 | 19.3 | 23.0 | 0.577 | 0.691 |
| 7 garden photos | 0.8 | 0.46 | 0.63 | 0.819 | 0.797 | 0.860 | 19.5 | 19.8 | 0.587 | 0.670 |
| 7 garden photos | 1.6 | 0.62 | 0.93 | 0.657 | 0.805 | 0.811 | 18.0 | 17.2 | 0.625 | 0.652 |
| 7 garden photos | 2.4 | 0.72 | 1.00 | 0.545 | 0.789 | 0.789 | 14.7 | 14.7 | 0.703 | 0.703 |
| 7 garden photos | 3.2 | 0.78 | 1.00 | 0.469 | 0.731 | 0.731 | 11.4 | 11.4 | 0.687 | 0.687 |
| synthetic streets | 0.4 | 0.26 | 0.25 | 0.948 | 0.690 | 0.944 | 18.7 | 25.8 | 0.958 | 0.961 |
| synthetic streets | 0.8 | 0.33 | 0.35 | 0.875 | 0.746 | 0.899 | 19.8 | 21.7 | 0.948 | 0.954 |
| synthetic streets | 1.6 | 0.44 | 0.57 | 0.754 | 0.793 | 0.837 | 19.4 | 17.4 | 0.938 | 0.941 |
| synthetic streets | 2.4 | 0.52 | 0.74 | 0.668 | 0.787 | 0.804 | 16.7 | 15.0 | 0.939 | 0.932 |
| synthetic streets | 3.2 | 0.58 | 0.87 | 0.605 | 0.770 | 0.773 | 14.2 | 13.3 | 0.936 | 0.930 |
| 9 real streets | 0.4 | 0.33 | 0.38 | 0.903 | 0.849 | 0.941 | 18.0 | 23.9 | 0.855 | 0.933 |
| 9 real streets | 0.8 | 0.43 | 0.55 | 0.806 | 0.864 | 0.907 | 18.6 | 20.9 | 0.868 | 0.918 |
| 9 real streets | 1.6 | 0.58 | 0.85 | 0.645 | 0.866 | 0.860 | 18.8 | 18.2 | 0.889 | 0.900 |
| 9 real streets | 2.4 | 0.67 | 0.99 | 0.531 | 0.841 | 0.839 | 17.5 | 17.4 | 0.894 | 0.894 |
| 9 real streets | 3.2 | 0.74 | 1.00 | 0.453 | 0.788 | 0.788 | 15.7 | 15.7 | 0.854 | 0.854 |

On the 7 **real** fog photos (section 1; real fog was not used for the fit), fixed against adaptive default: PSNR 15.91 / 15.48 dB, SSIM 0.706 / 0.718, edge F1 0.645 / 0.664.

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
| Full pipeline (Balanced default) | 0.706 | 0.645 | +1.11 | 0.793 | 0.938 | +1.16 | 0.866 | 0.889 | +0.48 | 0.725 |
| without histogram stretching | 0.766 | 0.689 | +0.64 | 0.795 | 0.940 | +1.15 | 0.870 | 0.903 | +0.43 | 0.725 |
| without frequency-domain enhancement | 0.729 | 0.688 | +0.86 | 0.824 | 0.941 | +0.83 | 0.892 | 0.918 | +0.36 | 0.732 |
| without adaptive smoothing + sharpening | 0.710 | 0.668 | +2.66 | 0.841 | 0.940 | +0.15 | 0.908 | 0.919 | +0.84 | 0.782 |
| without all three course modules | 0.783 | 0.491 | +0.90 | 0.871 | 0.943 | +0.02 | 0.906 | 0.871 | +0.46 | 0.839 |
| without the dark channel prior | 0.687 | 0.685 | +0.53 | 0.777 | 0.932 | +1.21 | 0.768 | 0.856 | +0.16 | 0.735 |
| without colour balance | 0.708 | 0.661 | +1.05 | 0.794 | 0.940 | +1.15 | 0.866 | 0.893 | +0.49 | 0.723 |
| without frequency, sharpening, smoothing and colour balance (Traffic preset) | 0.739 | 0.674 | +2.01 | 0.866 | 0.944 | +0.03 | 0.918 | 0.924 | +0.65 | 0.895 |

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
| Shipped default (tuned on these photos) | 0.713 | 15.77 |
| Tuned on the other 6 scenes, applied to the held-out one (60 random candidates) | 0.784 | 17.93 |
| Best of the 60 candidates for each scene separately (upper bound) | 0.789 | 17.78 |
| Unprocessed | 0.698 | 13.81 |

## 9. Video: synthetic traffic clip

30 frames, 640x400, fixed camera, 3 cars driving across, fog (beta 1.6) from the true depth and sensor noise (sigma 2). *Flicker* is the mean frame-to-frame change of the output in pixels that did not change in the input (static background); the clip is processed by `dehaze_video.TemporalDehazer`.

| Processing | Flicker (grey levels) | PSNR (dB) | SSIM | Seconds/frame |
| --- | ---: | ---: | ---: | ---: |
| Foggy input | 1.54 | 12.4 | 0.747 | - |
| Every frame on its own | 1.19 | 19.7 | 0.816 | 0.07 |
| Temporal (smoothed estimates + denoising) | 0.42 | 19.7 | 0.810 | 0.09 |
| Every frame on its own, strength from haze level | 0.86 | 17.0 | 0.831 | 0.11 |
| Temporal, strength from haze level | 0.34 | 16.9 | 0.828 | 0.12 |
