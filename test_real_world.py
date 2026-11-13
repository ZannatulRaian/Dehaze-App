"""Smoke tests for the unpaired real-world fog evaluation."""
import numpy as np
import evaluate
import dehaze_core as C

images = evaluate.load_unpaired_real_fog(320)
assert len(images) >= 15, f"Expected the bundled real fog set; found {len(images)}"
for name, image in images.items():
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3, name
    metrics = evaluate.unpaired_metrics(image, image)
    assert abs(metrics["visible_edge_gain_e"]) < 1e-9, name
    assert abs(metrics["gradient_ratio_r"] - 1.0) < 1e-6, name
    assert metrics["newly_clipped_share"] == 0.0, name
    output = C.dehaze_pipeline(image, C.auto_strength_params(image)[0])
    assert output.shape == image.shape and output.dtype == np.uint8, name
print(f"real-world validation smoke tests passed ({len(images)} images)")
