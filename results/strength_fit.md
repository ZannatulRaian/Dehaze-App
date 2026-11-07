Fit on all scene types: strength = clip(2.4 x haze - 0.45, 0.25, 1) (in dehaze_core: 2.1 and 0.35).

| Held-out scene type | Rule fit on the others | Score, fixed default | Score, adaptive | Score, per-image best | Light fog only (beta <= 0.8): fixed | adaptive |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 7 garden photos | 2.0 x haze - 0.30 | 1.210 | 1.252 | 1.283 | 1.276 | 1.411 |
| synthetic streets | 3.0 x haze - 0.95 | 1.215 | 1.301 | 1.342 | 1.198 | 1.513 |
| 9 real streets | 3.0 x haze - 0.60 | 1.294 | 1.337 | 1.418 | 1.314 | 1.446 |