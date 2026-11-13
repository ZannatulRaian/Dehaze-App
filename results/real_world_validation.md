# Real-world urban fog validation (unpaired images)

Images processed: **20** from `assets/test images/urban_real_foggy`. Maximum long edge: 800px.

These are real fog/smog photographs without aligned clear references. Therefore PSNR, SSIM, reference edge-F1, and true object-detection recall are deliberately not reported for this set.

## Mean no-reference indicators by method

| Method | Visible-edge gain e | Gradient ratio r | Newly clipped share | Contrast change | Entropy change | Processing (s/image) |
|---|---:|---:|---:|---:|---:|---:|
| Fog removal only (DCP) | +0.401 | 1.556 | 0.07% | +5.51 | +0.100 | 0.054 |
| Traffic monitoring | +0.770 | 2.060 | 0.84% | +18.46 | +0.301 | 0.008 |
| Balanced (haze-adaptive) | +0.721 | 2.749 | 0.55% | +16.50 | +0.545 | 0.208 |
| Course modules only | +0.519 | 2.527 | 1.20% | +17.91 | +0.437 | 0.054 |

## Interpretation and limitations

- Visible-edge gain and gradient ratio measure contrast/edge changes, not recovery of the true scene. Sharpening and noise can increase them.
- Entropy, contrast, and colourfulness are descriptive statistics; higher values are not automatically better.
- Newly clipped share is a warning indicator for lost shadow/highlight detail.
- The human-review template is intentionally blank. Complete it after independent visual review; do not treat generated scores as human ratings.
- For stronger field validation, collect repeated daytime/nighttime samples across light, moderate, and dense fog; record capture conditions and source/licence; preserve original files; and obtain aligned clear references where feasible.
- For traffic claims, annotate vehicles/pedestrians and calculate precision, recall, and mAP against those annotations. Detector agreement with another model is not ground truth.

## Output files

- `real_world_unpaired.csv`: per-image, per-method metrics and runtime.
- `human_review_template.csv`: blank reviewer metadata and scoring sheet.
- `figures/real_world_unpaired_contact_sheet.jpg`: originals alongside haze-adaptive outputs.
