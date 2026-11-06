# Evaluation results

7 scenes, photos scaled to at most 800px on the long edge. PSNR and SSIM are measured against the clear reference photo of each scene; the contrast gain is relative to the foggy input and edge pixels are the share of pixels with a gradient above 12. Each value is the mean over the scenes.

## 1. Real fog

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 13.7 | 0.674 | 1.00 | 13.0 | 0.00 |
| Histogram stretching only | 12.1 | 0.701 | 2.43 | 35.4 | 0.01 |
| Frequency-domain enhancement only | 13.8 | 0.739 | 1.08 | 25.7 | 0.05 |
| Adaptive smoothing + sharpening only | 13.7 | 0.602 | 1.01 | 8.3 | 0.04 |
| The three course modules together | 13.7 | 0.684 | 2.07 | 30.8 | 0.08 |
| Dark channel prior only | 17.0 | 0.769 | 1.17 | 22.7 | 0.09 |
| Full pipeline (Balanced default) | 15.9 | 0.706 | 2.03 | 34.6 | 0.10 |

## 2. Simulated: fog

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.2 | 0.657 | 1.00 | 15.5 | 0.00 |
| Histogram stretching only | 10.5 | 0.765 | 2.25 | 37.4 | 0.01 |
| Frequency-domain enhancement only | 10.2 | 0.732 | 1.07 | 29.4 | 0.05 |
| Adaptive smoothing + sharpening only | 10.2 | 0.569 | 1.01 | 9.9 | 0.05 |
| The three course modules together | 11.0 | 0.694 | 1.99 | 34.2 | 0.09 |
| Dark channel prior only | 21.5 | 0.943 | 1.25 | 36.0 | 0.10 |
| Full pipeline (Balanced default) | 18.0 | 0.805 | 1.91 | 47.0 | 0.11 |

## 2. Simulated: fog + Gaussian blur (2 px)

| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foggy input (no processing) | 10.1 | 0.411 | 1.00 | 0.1 | 0.00 |
| Histogram stretching only | 10.2 | 0.485 | 2.44 | 2.6 | 0.01 |
| Frequency-domain enhancement only | 10.1 | 0.433 | 1.01 | 0.2 | 0.04 |
| Adaptive smoothing + sharpening only | 10.1 | 0.399 | 1.00 | 0.1 | 0.04 |
| The three course modules together | 10.8 | 0.492 | 2.23 | 4.7 | 0.08 |
| Dark channel prior only | 19.1 | 0.573 | 1.24 | 1.3 | 0.09 |
| Full pipeline (Balanced default) | 16.4 | 0.578 | 1.95 | 8.1 | 0.09 |
| Full pipeline + Wiener deblur (blur known) | 17.2 | 0.670 | 1.91 | 21.5 | 0.13 |

## 2. Simulated: fog + horizontal motion blur (2 px)

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
