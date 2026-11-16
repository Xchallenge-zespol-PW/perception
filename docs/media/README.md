# Media for the README

Showcase assets for the [root README](../../README.md). Keep rosbag / raw `data/` offline — they are training sources, not GitHub demos.

## `demo/` — competition / robot visuals

| File | Purpose | Guidance |
|------|---------|----------|
| `sorting_demo.gif` (or `.mp4`) | Main CV signal | Prefer **5–10 s** of the robot arranging cubes (masks/poses visible if you have overlay). Looping GIF is ideal for GitHub. |
| `robot.jpg` | Hero photo | One clean still of the Unitree G1 (or setup). Avoid cluttered multi-panel collages. |

TODO: *Add `docs/media/demo/sorting_demo.gif` (5–10 s cube-sorting clip from competition footage).*

TODO: *Add `docs/media/demo/robot.jpg` (clean photo of the G1 / setup).*

Optional extras (link from README only if useful):

- short unlisted YouTube / Drive clip of a full competition run
- one debug-image still (`/perception/debug_image`) showing masks + slot IDs

Do **not** commit MCAP bags or long raw recordings into this folder.

## `training/` — Ultralytics exports

Copy plots from the training machine (Colab / other GPU host), e.g.:

| File | Typical Ultralytics source |
|------|----------------------------|
| `results.png` | `runs/seg/.../results.png` |
| `confusion_matrix.png` | optional |
| `BoxPR_curve.png` / `MaskPR_curve.png` | optional |

TODO: *Copy `results.png` from the training host into `docs/media/training/`.*

TODO: *Paste segmentation mAP (and optional notes) into the root README Results table.*

Latency numbers for TensorRT vs ONNX are already in the root README.

## Naming checklist

```text
docs/media/demo/sorting_demo.gif
docs/media/demo/robot.jpg
docs/media/training/results.png
```

Keep filenames stable so README embeds keep working.
