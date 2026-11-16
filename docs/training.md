# Data and models: seg + pose

End-to-end flow from RealSense bags → CVAT labels → `cubes_seg` / `cubes_pose` → Ultralytics train → export `.onnx` / TensorRT `.engine` → robot. Training usually runs on a GPU host (e.g. Colab); the laptop handles conversion and CPU benches.

Flow for both segmenters: `yolo_seg` (full-cube mask) and `yolo_pose` (4 top-face corners `c0..c3`). One labeling pass produces both datasets.

```text
record_cubes.sh → extract_frames.py → CVAT (polygon + 4 corners)
  → cvat_to_yolo.py → cubes_seg/ + cubes_pose/ → preview_pose_labels.py
  → train_yolo.py (Colab GPU) → export_model.py (.pt + .onnx)
  → deploy_robot.sh → export_model.py --formats engine (on Jetson) → compare_backends.py → bench_models.py
```

Laptop has no CUDA (AMD), so training runs in Colab; on the laptop: conversion, preview, `.onnx` export, and CPU bench.

## 0. Before training: will YOLO26 run on the Jetson?

In the humble container on the robot, after enabling GPU. Paths like `/ws/src/perception/...` are the **on-robot tree after** `deploy_robot.sh` (this repo keeps tools at `/ws/tools` and the package at `/ws/src/g1_perception`).

```bash
YOLO_AUTOINSTALL=false python3 /ws/src/perception/tools/jetson_gpu_check.py \
  --export yolo26n-pose.pt --workdir /ws/models/check
```

All `PASS` lines = environment ready and YOLO26 → TensorRT export works. If `engine export` is `FAIL`, retry with `--export yolo11n-pose.pt` and train on `yolo11*` weights (`--model yolo11s-pose.pt`).

## 1. Recordings

To compare `pnp` vs `geometric` you need **known yaw** per cube (angle mat):

```bash
./tools/record_cubes.sh RGGR mocne 15 "" "0,15,45,30"
python3 tools/extract_frames.py
```

Vary lighting (`dzienne`, `mocne`, `niskie`, `ciemnenoise`) and layouts; also touching cubes and 30–45° rotations. Frames land in `data/frames/<bag>/<n>_color.png`.

## 2. Labeling in CVAT

Project with labels **`red`** and **`green`** (other names map via `config/labels.yaml`, e.g. `czerwona`). Task: upload frames **with recording folders** (zip of `data/frames/`), because train/val split is by recording = first directory in the image path.

Per cube:

| Shape | What to mark |
|-------|--------------|
| **Polygon** | outline of the whole visible cube (top + sides) → seg mask and pose box |
| **Points** (4 points) or **Skeleton** with 4 points | **top-face** corners, any order |

- Point order does not matter: the converter sorts them to `c0..c3` (visually clockwise, `c0` = lowest; if `v` differs by < 3 px, the righter one).
- Points attach to a polygon via a shared **group** (Ctrl+G) or, without a group, to the smallest polygon containing their centroid. Group touching cubes.
- Occluded corner whose location is still known: keep the point and set `occluded` (visibility 1). Impossible corner: cube without points (skeleton: `outside`) — pose gets all zeros.
- Cube whose top face is not visible: polygon only.

Export: **CVAT for images 1.1** (XML).

## 3. Convert and check

```bash
python3 tools/cvat_to_yolo.py annotations.xml --images-root data/frames
# or with fixed validation recordings:
python3 tools/cvat_to_yolo.py annotations.xml --images-root data/frames \
  --val-groups kostki_RGGR_niskie_20261015_184332 kostki_RGRG_ciemnenoise_20261015_185122
python3 tools/preview_pose_labels.py datasets/cubes_pose --split train --limit 40
```

- Output: `datasets/cubes_seg/` and `datasets/cubes_pose/` (`images/`, `labels/`, `data.yaml` with `kpt_shape: [4, 3]`, `flip_idx: [0, 3, 2, 1]`). An existing directory is not overwritten without `--overwrite` (move an old Roboflow dataset or use `--seg-name`).
- Review `WARN` lines (points outside polygon, unknown label, missing image) before training.
- `preview_pose_labels.py` exits 1 if any label breaks `c0..c3` order (e.g. dataset labeled outside CVAT), and writes numbered-corner previews to `<dataset>/preview_<split>/`.

## 4. Training (Colab GPU)

```bash
pip install -q -c deploy/constraints.txt ultralytics
python3 tools/train_yolo.py --task seg --device 0
python3 tools/train_yolo.py --task pose --device 0
# larger model when robot bench allows:
python3 tools/train_yolo.py --task pose --model yolo26s-pose.pt --device 0
```

- Defaults: `yolo26n-seg.pt` / `yolo26n-pose.pt`, 100 epochs, `imgsz=640`, `hsv_h=0`; pose with `fliplr=0` (flip breaks the `c0` rule).
- `ultralytics==8.4.191` from `deploy/constraints.txt` — same version in Colab, on the laptop, and on the robot.
- Weights: `runs/seg/cubes/weights/best.pt`, `runs/pose/cubes_pose/weights/best.pt`.

## 5. Export and deploy

On the laptop (or in Colab):

```bash
python3 tools/export_model.py runs/seg/cubes/weights/best.pt --name cubes_seg
python3 tools/export_model.py runs/pose/cubes_pose/weights/best.pt --name cubes_pose
./tools/deploy_robot.sh models/cubes_seg.pt models/cubes_seg.onnx \
  models/cubes_pose.pt models/cubes_pose.onnx
```

On the robot (in the container), TensorRT engines from the uploaded `.pt` — `.onnx` (CPU fallback) stays untouched:

```bash
cd /ws
python3 src/perception/tools/export_model.py /ws/models/cubes_seg.pt --formats engine --out /ws/models
python3 src/perception/tools/export_model.py /ws/models/cubes_pose.pt --formats engine --out /ws/models
# engine (GPU, FP16) vs .onnx (CPU) on the same frames — should be OK:
python3 src/perception/tools/compare_backends.py /ws/models/cubes_pose.engine /ws/models/cubes_pose.onnx --frames /ws/data/frames
```

`params_robot.yaml` loads `/ws/models/cubes_seg.engine` on GPU with fallback `/ws/models/cubes_seg.onnx` on CPU ([robot.md](robot.md)).

## 6. Bench and size choice

```bash
# robot: GPU (.engine) and CPU fallback (.onnx) on real frames
python3 src/perception/tools/bench_models.py /ws/models/cubes_pose.engine \
  /ws/models/cubes_pose.onnx --device 0 --frames /ws/data/frames --csv /ws/results/bench_robot.csv
# laptop: CPU
python3 tools/bench_models.py models/cubes_pose.onnx models/cubes_seg.onnx --device cpu
```

Pick the largest size (`n` / `s` / `m`) that on the robot GPU fits under the 15 fps camera budget (~66 ms) with headroom, and on CPU fallback still yields usable voting. Record the result in the table in [robot.md](robot.md).
