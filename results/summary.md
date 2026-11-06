# Evaluation results

7 real scenes, photos scaled to at most 800px on the long edge. PSNR and SSIM are measured against the clear reference photo of each scene; the contrast gain is relative to the foggy input and edge pixels are the share of pixels with a gradient above 12. Each value is the mean over the scenes. The downstream test runs a standard Canny edge detector (fixed thresholds 50/150, which is what makes fog costly) and a probabilistic Hough line detector on every result: *Edge F1* compares its edges with the clear photo's (2 px tolerance); *auto thresholds* repeats that with Canny thresholds set from each image's median brightness, which compensates for low contrast and so shows the gain that is not just a brighter or higher-contrast picture; *line recall* is the share of the clear photo's lines found again; *vertical* lines are within 15 degrees of vertical (building edges) and *oblique* lines are 20-70 degrees from horizontal (lane and roof lines). More edges or lines is not better by itself: noise and over-sharpening add false ones, and the Hough line count also depends on how segments merge, so compare with F1 and recall.

## 1. Real fog

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 13.7 | 0.674 | 1.00 | 13.0 | 0.00 |
| Histogram stretching only | 12.1 | 0.701 | 2.43 | 35.4 | 0.01 |
| Frequency-domain enhancement only | 13.8 | 0.739 | 1.08 | 25.7 | 0.03 |
| Adaptive smoothing + sharpening only | 13.7 | 0.602 | 1.01 | 8.3 | 0.03 |
| The three course modules together | 13.7 | 0.684 | 2.07 | 30.8 | 0.06 |
| Dark channel prior only | 17.0 | 0.769 | 1.17 | 22.7 | 0.06 |
| Full pipeline (Balanced default) | 15.9 | 0.706 | 2.03 | 34.6 | 0.07 |

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

## 2. Simulated: fog

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.2 | 0.657 | 1.00 | 15.5 | 0.00 |
| Histogram stretching only | 10.5 | 0.765 | 2.25 | 37.4 | 0.01 |
| Frequency-domain enhancement only | 10.2 | 0.732 | 1.07 | 29.4 | 0.03 |
| Adaptive smoothing + sharpening only | 10.2 | 0.569 | 1.01 | 9.9 | 0.03 |
| The three course modules together | 11.0 | 0.694 | 1.99 | 34.2 | 0.06 |
| Dark channel prior only | 21.5 | 0.943 | 1.25 | 36.0 | 0.06 |
| Full pipeline (Balanced default) | 18.0 | 0.805 | 1.91 | 47.0 | 0.07 |

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

## 2. Simulated: fog + Gaussian blur (2 px)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.1 | 0.411 | 1.00 | 0.1 | 0.00 |
| Histogram stretching only | 10.2 | 0.485 | 2.44 | 2.6 | 0.01 |
| Frequency-domain enhancement only | 10.1 | 0.433 | 1.01 | 0.2 | 0.03 |
| Adaptive smoothing + sharpening only | 10.1 | 0.399 | 1.00 | 0.1 | 0.03 |
| The three course modules together | 10.8 | 0.492 | 2.23 | 4.7 | 0.06 |
| Dark channel prior only | 19.1 | 0.573 | 1.24 | 1.3 | 0.06 |
| Full pipeline (Balanced default) | 16.4 | 0.578 | 1.95 | 8.1 | 0.07 |
| Full pipeline + Wiener deblur (blur known) | 17.2 | 0.670 | 1.91 | 21.5 | 0.09 |
| Full pipeline + Wiener deblur (blur estimated) | 17.2 | 0.666 | 1.90 | 20.4 | 0.12 |

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
| Full pipeline + Wiener deblur (blur known) | 8.2 | 0.739 | 0.791 | 524 | 51 | 207 | 0.600 | 7/7 |
| Full pipeline + Wiener deblur (blur estimated) | 7.6 | 0.746 | 0.795 | 430 | 39 | 174 | 0.576 | 7/7 |

## 2. Simulated: fog + horizontal motion blur (2 px)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.1 | 0.478 | 1.00 | 3.0 | 0.00 |
| Histogram stretching only | 10.3 | 0.564 | 2.37 | 17.0 | 0.01 |
| Frequency-domain enhancement only | 10.1 | 0.519 | 1.03 | 9.1 | 0.03 |
| Adaptive smoothing + sharpening only | 10.1 | 0.444 | 1.00 | 1.9 | 0.03 |
| The three course modules together | 10.7 | 0.555 | 2.14 | 16.8 | 0.06 |
| Dark channel prior only | 19.9 | 0.681 | 1.24 | 13.3 | 0.06 |
| Full pipeline (Balanced default) | 17.2 | 0.655 | 1.92 | 25.7 | 0.07 |
| Full pipeline + Wiener deblur (blur known) | 17.9 | 0.782 | 1.93 | 44.4 | 0.09 |
| Full pipeline + Wiener deblur (blur estimated) | 17.6 | 0.738 | 1.92 | 44.3 | 0.11 |

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
| Full pipeline + Wiener deblur (blur known) | 10.8 | 0.654 | 0.750 | 842 | 76 | 338 | 0.606 | 7/7 |
| Full pipeline + Wiener deblur (blur estimated) | 10.5 | 0.655 | 0.751 | 809 | 78 | 315 | 0.608 | 7/7 |

## 3. Urban test set: synthetic city streets with true depth

6 synthetic streets (assets/test images/urban_synthetic, 640x400, different building layouts, cars, lane markings), fogged with t = exp(-beta * depth) from their true depth map.

### 3.1. Urban: fog (beta 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 12.2 | 0.754 | 1.00 | 6.9 | 0.00 |
| Histogram stretching only | 12.5 | 0.773 | 1.56 | 7.1 | 0.00 |
| Frequency-domain enhancement only | 12.1 | 0.750 | 1.10 | 9.1 | 0.01 |
| Adaptive smoothing + sharpening only | 12.1 | 0.742 | 1.11 | 13.7 | 0.02 |
| The three course modules together | 12.2 | 0.696 | 1.31 | 24.7 | 0.03 |
| Dark channel prior only | 21.8 | 0.871 | 1.08 | 7.1 | 0.03 |
| Full pipeline (Balanced default) | 19.4 | 0.793 | 1.29 | 16.4 | 0.04 |

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

### 3.2. Urban: fog + horizontal motion blur (2 px) (beta 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 12.0 | 0.647 | 1.00 | 6.7 | 0.00 |
| Histogram stretching only | 11.3 | 0.664 | 1.69 | 16.8 | 0.00 |
| Frequency-domain enhancement only | 12.0 | 0.659 | 1.04 | 16.0 | 0.01 |
| Adaptive smoothing + sharpening only | 12.0 | 0.662 | 1.04 | 12.9 | 0.02 |
| The three course modules together | 11.9 | 0.663 | 1.45 | 25.7 | 0.03 |
| Dark channel prior only | 20.7 | 0.776 | 1.00 | 16.5 | 0.03 |
| Full pipeline (Balanced default) | 18.8 | 0.752 | 1.29 | 22.2 | 0.04 |
| Full pipeline + Wiener deblur (blur known) | 19.4 | 0.756 | 1.37 | 22.3 | 0.04 |
| Full pipeline + Wiener deblur (blur estimated) | 19.2 | 0.742 | 1.37 | 23.1 | 0.05 |

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
| Full pipeline + Wiener deblur (blur known) | 6.2 | 0.920 | 0.883 | 136 | 28 | 16 | 0.751 | 6/6 |
| Full pipeline + Wiener deblur (blur estimated) | 6.1 | 0.919 | 0.881 | 137 | 28 | 16 | 0.750 | 6/6 |

### 3.3. Urban: fog + Gaussian blur (2 px) (beta 1.6)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 11.9 | 0.609 | 1.00 | 8.5 | 0.00 |
| Histogram stretching only | 11.2 | 0.636 | 1.71 | 18.9 | 0.00 |
| Frequency-domain enhancement only | 11.9 | 0.632 | 1.02 | 11.9 | 0.01 |
| Adaptive smoothing + sharpening only | 11.9 | 0.631 | 1.01 | 10.0 | 0.02 |
| The three course modules together | 12.3 | 0.707 | 1.62 | 17.7 | 0.03 |
| Dark channel prior only | 20.3 | 0.747 | 0.96 | 14.3 | 0.03 |
| Full pipeline (Balanced default) | 19.5 | 0.772 | 1.22 | 18.1 | 0.03 |
| Full pipeline + Wiener deblur (blur known) | 19.2 | 0.777 | 1.35 | 21.2 | 0.04 |
| Full pipeline + Wiener deblur (blur estimated) | 19.1 | 0.785 | 1.36 | 20.3 | 0.05 |

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
| Full pipeline + Wiener deblur (blur known) | 5.9 | 0.937 | 0.933 | 128 | 23 | 19 | 0.715 | 6/6 |
| Full pipeline + Wiener deblur (blur estimated) | 5.8 | 0.941 | 0.949 | 121 | 30 | 15 | 0.738 | 6/6 |

### Fog density sweep (urban set, full pipeline vs doing nothing)

| Beta | Input PSNR | Full PSNR | Input SSIM | Full SSIM | Input edge F1 | Full edge F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8 | 16.6 | 19.8 | 0.875 | 0.746 | 0.952 | 0.948 |
| 1.6 | 12.2 | 19.4 | 0.754 | 0.793 | 0.836 | 0.938 |
| 2.4 | 10.0 | 16.7 | 0.668 | 0.787 | 0.804 | 0.939 |
| 3.2 | 8.7 | 14.2 | 0.605 | 0.770 | 0.801 | 0.936 |

### Transmission estimate against the true depth

Mean over the urban set; the true transmission is exp(-beta * depth), clipped to the same minimum as the estimate.

| Beta | Transmission MAE | Correlation with true t | Airlight error (0-255) |
| ---: | ---: | ---: | ---: |
| 0.8 | 0.094 | 0.908 | 5.7 |
| 1.6 | 0.070 | 0.939 | 5.0 |
| 2.4 | 0.060 | 0.951 | 4.7 |
| 3.2 | 0.056 | 0.959 | 4.0 |

## 4. Blur estimation

`estimate_blur` measures the width of the strongest edges across and along the image and subtracts the width of a sharp photo. Each row blurs the clear photos by a known amount (fog added first, beta 1.6 unless the row says otherwise) and compares the estimate with it. The Gaussian calibration factor was fit on the real photos, so the synthetic cities are an independent check. Motion blur means horizontal motion, length 3 x amount.

| True blur | Fog beta | Real photos: estimate (px) | kind right | abs. error (px) | Synthetic cities: estimate (px) | kind right | abs. error (px) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none (sharp) | 0 | 0.00 | 100% | 0.00 | 0.00 | 100% | 0.00 |
| none (sharp) | 1.6 | 0.00 | 100% | 0.00 | 0.00 | 100% | 0.00 |
| none (sharp) | 3 | 0.00 | 100% | 0.00 | 0.00 | 100% | 0.00 |
| Gaussian 1 px | 1.6 | 0.81 | 86% | 0.23 | 1.18 | 100% | 0.18 |
| Gaussian 2 px | 1.6 | 1.97 | 100% | 0.20 | 1.98 | 100% | 0.03 |
| Gaussian 3 px | 1.6 | 3.07 | 100% | 0.26 | 2.68 | 100% | 0.32 |
| Gaussian 4 px | 1.6 | 3.95 | 100% | 0.24 | 3.50 | 100% | 0.50 |
| Motion 1 px | 1.6 | 0.25 | 29% | 0.75 | 0.93 | 100% | 0.07 |
| Motion 2 px | 1.6 | 2.09 | 100% | 0.17 | 1.71 | 100% | 0.29 |
| Motion 3 px | 1.6 | 2.65 | 100% | 0.35 | 2.20 | 100% | 0.80 |
| Motion 4 px | 1.6 | 3.48 | 100% | 0.57 | 3.29 | 100% | 0.71 |
| Gaussian 2 px | 3 | 1.87 | 100% | 0.20 | 1.96 | 100% | 0.04 |
| Motion 2 px | 3 | 1.97 | 100% | 0.12 | 1.73 | 100% | 0.27 |

On the 7 real hazy photos (true blur unknown) the estimator found: none 7 (largest estimate 0.00 px).
