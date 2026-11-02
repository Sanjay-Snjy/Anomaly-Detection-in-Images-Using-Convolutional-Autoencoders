"""Recalibrate the stale BatchNorm statistics of the saved screw autoencoder.

The conv weights were trained with batch statistics, but the moving statistics
stored in the .keras file never converged, so at inference every activation is
divided by a variance ~300x too large and the output collapses to one gray value.
Here we re-estimate mean/variance from the normal training images and measure
whether the reconstruction (and the anomaly scores) become usable.
"""

import numpy as np
import keras

from anomaly_detection import (
    category_paths,
    load_image,
    load_model,
    load_test_images,
    score_images,
)

paths = category_paths("screw")
model = load_model(paths["model"])

train_paths = sorted(paths["train_good"].glob("*.png"))
train = np.stack([load_image(q) for q in train_paths]).astype(np.float32)
print(f"training images: {len(train)}")

probe = load_image(sorted((paths["test"] / "good").glob("*.png"))[0])


def out_stats(tag):
    out = model.predict(probe[None], verbose=0)[0]
    print(f"{tag}: output mean {out.mean():.4f} std {out.std():.4f} "
          f"min {out.min():.3f} max {out.max():.3f}")


out_stats("before")

bn_layers = [l for l in model.layers if isinstance(l, keras.layers.BatchNormalization)]
feeder = keras.Model(model.input, [l.input for l in bn_layers])

sums = [None] * len(bn_layers)
sqsums = [None] * len(bn_layers)
counts = [0] * len(bn_layers)

for start in range(0, len(train), 16):
    acts = feeder(train[start:start + 16], training=True)
    for j, act in enumerate(acts):
        flat = act.numpy().reshape(-1, act.shape[-1]).astype(np.float64)
        if sums[j] is None:
            sums[j] = flat.sum(axis=0)
            sqsums[j] = (flat * flat).sum(axis=0)
        else:
            sums[j] += flat.sum(axis=0)
            sqsums[j] += (flat * flat).sum(axis=0)
        counts[j] += flat.shape[0]

for j, layer in enumerate(bn_layers):
    mean = sums[j] / counts[j]
    var = sqsums[j] / counts[j] - mean ** 2
    layer.moving_mean.assign(mean.astype("float32"))
    layer.moving_variance.assign(np.maximum(var, 1e-6).astype("float32"))
    print(f"{layer.name:24s} new moving_var mean {np.maximum(var, 1e-6).mean():.5f}")

out_stats("after")

# Does it still separate normals from defects?
x_test, y_test, types, _ = load_test_images("screw")
scores, _ = score_images(model, x_test)
normal = scores[y_test == 0]
defect = scores[y_test == 1]
threshold = normal.mean() + 3 * normal.std()
tp = int((defect > threshold).sum())
print(f"auc      {__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(y_test, scores):.4f}")
print(f"threshold {threshold:.5f}  recall {tp}/{len(defect)}  false positives {int((normal > threshold).sum())}/{len(normal)}")
print(f"score normal {normal.mean():.5f}±{normal.std():.5f}  defect {defect.mean():.5f}")

model.save("models/screw_autoencoder.recalibrated.keras")
print("saved models/screw_autoencoder.recalibrated.keras")
