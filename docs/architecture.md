# Architecture

RGB-D cube perception: detect a cube at known table slots (red / green / empty, ~5 cm cube) and estimate its pose (3D position + orientation). Core logic (`core/`, `segmenters/`, `pose/`, `pipeline`, `node_config`) does not import ROS; ROS 2 nodes are a thin wrapper.

## Pipeline

```text
Frame (RGB + depth + K)
        │
        ▼
   Segmenter  →  list[Detection]     # "hsv" | "yolo_seg"
        │
        ▼
 PoseEstimator  →  Pose | None       # "centroid" | "geometric" | "pnp"
        │
        ▼
 list[SlotResult]                    # red | green | empty | uncertain
```

| Layer | What it provides |
|-------|------------------|
| `core.types` | `Frame`, `Detection`, `Pose`, `SlotResult`, label constants, protocols |
| `core.config` | ROI / HSV / label map from YAML |
| `core.labels` | `canonical_label` (model class name → canonical label) |
| `core.voting` | `SlotVoter`: per-slot voting over a time window (seconds) or N frames |
| `core.tracker` | 3D tracking (nearest neighbour) + per-track voting |
| `core.slot_assign` | slot numbers 1..N along an axis + line/spacing validation |
| `core.cube_registry` | exercise 1: freeze start layout, stable `cube_N` names, re-find lost cubes |
| `core.dataset` | `load_frame` from `data/index.csv` |
| `core.geometry` | deprojection, plane RANSAC, cube pose, `transform_pose` |
| `core.keypoints` | canonical order of the 4 top-face corners (`c0..c3`) |
| `core.pnp` | top-face pose from 4 corners (IPPE_SQUARE), pick solution by table normal, yaw on table, body center |
| `core.frame_utils` | depth mm → m, `K` from `camera_info` |
| `core.debug_vis` | debug image (ROI / track id, masks, labels, yaw), mono8 mask sum |
| `pipeline` | `@register`, `Pipeline.from_config` / `process` (`rois_mode`) |
| `segmenters/` | `hsv`, `yolo_seg` |
| `pose/` | `centroid`, `geometric` |
| `node_config` | node params → `Pipeline` / `SlotVoter` config (no ROS) |
| `ros_conversions` | `VotedSlot` → `vision_msgs/Detection3DArray` |
| `perception_node` | thin rclpy node (sync, cv_bridge, TF, publish) |
| `tools/` | record → frames → calibrate → autolabel → train |

Planned: `evaluate`, FoundationPose. TensorRT export (`.engine`) is already supported in the loader.

## Package layout

```text
src/g1_perception/
├── config/                      # rois, labels, params_{laptop,robot}.yaml
├── launch/
│   ├── perception.launch.py
│   └── apriltag.launch.py
├── g1_perception/
│   ├── core/                    # types, geometry, voting, tracker, cube_registry, …
│   ├── segmenters/              # hsv, yolo_seg
│   ├── pose/                    # centroid, geometric, pnp
│   ├── pipeline.py
│   ├── node_config.py
│   ├── ros_conversions.py
│   ├── perception_node.py
│   └── apriltag_node.py
└── test/
```

Related top-level dirs: `tools/`, `docker/`, `deploy/`, `docs/`, `models/` (weights). Local artifacts (`data/`, `datasets/`, `runs/`, bags under `/bags/`) are gitignored — see [environment.md](environment.md).

## Python modules

### `core.types`

| Type | Fields |
|------|--------|
| `Frame` | `rgb` HxWx3 uint8, `depth_m` HxW float32, `K` 3×3, `stamp` |
| `Detection` | `slot`, `label` (raw segmenter name), `score`, `mask` HxW bool, optional `keypoints` Kx3 `(u, v, conf)` (corners `c0..c3` from `yolo_pose`) |
| `Pose` | `position` (3,), `quat` xyzw (4,), `score`, optional `method` (`pnp` \| `fallback:<reason>`) |
| `SlotResult` | `slot`, `label` (canonical), `score`, `detection \| None`, `pose \| None` |
| `Segmenter` | `segment(frame) -> list[Detection]` |
| `PoseEstimator` | `estimate(frame, det) -> Pose \| None` |

Canonical labels (constants): `LABEL_RED='red'`, `LABEL_GREEN='green'`, `LABEL_EMPTY='empty'`, `LABEL_UNCERTAIN='uncertain'`.

### `core.dataset`

```python
from g1_perception.core.dataset import load_frame
frame = load_frame(row)  # row from data/index.csv
```

### `core.config`

```python
from g1_perception.core.config import load_rois, load_hsv, load_label_map
rois = load_rois()
hsv = load_hsv()
label_map = load_label_map()  # {'red cube': 'red', 'zielony': 'green', ...}
```

### `core.geometry`

| Function | Description |
|----------|-------------|
| `deproject(mask, depth_m, K)` | Nx3 points (skips zeros; optional Z range) |
| `quad_mask(corners, shape)` | HxW bool filled polygon of corners (Nx2 or Kx3); for `yolo_pose` top face only |
| `fit_plane_ransac(points)` | `(normal, d, n_inliers)` |
| `table_plane_around(roi, mask, …)` | table plane around ROI excluding object mask |
| `table_plane_around_rois(rois, exclude, …)` | one table plane around many ROIs excluding `exclude` pixels (all detection masks) |
| `cube_pose_on_table(points, plane, size=0.05)` | `Pose` (minAreaRect, yaw % 90°, center + size/2 along normal) |

### `pipeline`

```python
import g1_perception.segmenters  # registers hsv, yolo_seg
import g1_perception.pose        # registers centroid, geometric
from g1_perception.pipeline import Pipeline

pipe = Pipeline.from_config({
    # optional; defaults config/rois.yaml and config/labels.yaml
    # 'rois_path': '...', 'labels_path': '...',
    'segmenter': {
        'name': 'hsv',
        'params': {'min_frac': 0.05, 'erode': 1},
    },
    'pose_estimator': {
        'name': 'geometric',
        'params': {'size': 0.05, 'margin': 20},
    },
})
# or YOLO:
# 'segmenter': {'name': 'yolo_seg', 'params': {
#     'model_path': 'runs/seg/cubes/weights/best.pt',
#     'verify_hsv': True,
# }}
results = pipe.process(frame)  # list[SlotResult], one per ROI, sorted by slot
```

`process` rules:

- exactly one `SlotResult` per ROI from `rois.yaml`;
- no detection in ROI → `empty` (`score=0`, `detection=None`);
- several detections in one slot → highest `score` wins; detections without a slot are dropped;
- label mapped via `labels.yaml`; unknown name → `uncertain`;
- pose computed only for `red` / `green` (`pose=None` for `empty` / `uncertain`).

`from_config` loads ROI and label map once and passes them into stages whose constructor accepts `rois` / `label_map` (unless given explicitly in `params` or via `rois_path` / `labels_path`).

### `core.voting`

```python
from g1_perception.core.voting import SlotVoter

voter = SlotVoter(window=10, min_ratio=0.7)  # min_frames defaults to window
for frame in frames:
    voted = voter.update(pipe.process(frame))  # list[VotedSlot], per slot

# time window: same decisions at 1 and 20 fps
voter = SlotVoter(window=100, min_ratio=0.7, window_s=2.0, min_votes=5)
voted = voter.update(pipe.process(frame), frame.stamp)
```

Per slot over the last `window` frames, or with `window_s` over votes not older than `window_s` from the newest frame stamp (requires `min_votes`; a backward stamp, e.g. `ros2 bag play --loop`, clears history):

- label = majority; `uncertain` when share &lt; `min_ratio`, tie, too few frames (&lt; `min_frames`), or winner is `uncertain`; `uncertain` frames count in the denominator;
- for `red` / `green`: position = per-axis median of frames matching the winner; yaw = median with 90° symmetry (circular mean of `4·yaw` as reference, residuals wrapped to ±45°, then median); Z axis = mean table normal;
- `VotedSlot`: `slot`, `label`, `score` (winner share), `n_frames`, `pose`, `yaw_deg` ∈ [0, 90), `yaw_consistency` ∈ [0, 1] (1 = consistent angles).

### Segmenters

| Name | Module | Behavior |
|------|--------|----------|
| `hsv` | `segmenters/hsv.py` | Per ROI: HSV red/green masks, erode, fraction ≥ `min_frac` → `red`/`green`; below threshold → no detection (→ `empty`) |
| `yolo_seg` | `segmenters/yolo_seg.py` | YOLO-seg (`.pt`/`.onnx`/`.engine`, load via `segmenters/yolo_backend.py` with CPU fallback; `.engine` without torch via `trt_runtime.py` + `yolo_decode.py`), `retina_masks=True`, slot when mask∩ROI coverage &gt; 50%; keeps model class name; optional `verify_hsv` (HSV mismatch → `uncertain`) |

`ultralytics` is imported only inside `YoloSegSegmenter` (optional dependency).

### Pose estimators

| Name | Module | Behavior |
|------|--------|----------|
| `centroid` | `pose/centroid.py` | Median of 3D points; orientation from table normal |
| `geometric` | `pose/geometric.py` | `cube_pose_on_table` (table RANSAC + minAreaRect); `estimate_many` (pipeline `rois_mode=none`): all frame detections on one table plane excluding their masks |
| `pnp` | `pose/pnp.py` | orientation from 4 corners (IPPE_SQUARE, project onto table), position from depth in the quad; reject (`no_keypoints`, `kpt_conf`, `no_table`, `no_depth`, `pnp_failed`, `reproj`, `tilt`, `depth_diff`) → `geometric` on top face or `None` |

## Unit tests (map)

| File | Scope |
|------|-------|
| `test_pipeline.py` | registry, `from_config`, 1 result/ROI (4 ROI + 2 YOLO dets → 2 colors + 2 `empty`), `uncertain` |
| `test_labels.py` | `canonical_label`, `load_label_map`, default `labels.yaml` |
| `test_voting.py` | `SlotVoter` (majority, `min_ratio`, tie, window), position median, yaw across 0/90 |
| `test_cube_registry.py` | freeze layout, track pushed cube, re-find after loss, color veto, reset |
| `test_frame_utils.py` | depth mm → m, `32FC1` NaN → 0, `K`, `transform_pose` |
| `test_debug_vis.py` | mono8 mask sum, debug image |
| `test_node_config.py` | node params → pipeline / voter config, HSV run |
| `test_ros_conversions.py` | `Detection3DArray` contract (skipped without `vision_msgs`) |
| `test_dataset.py` | `load_frame` |
| `test_config.py` | ROI / HSV YAML |
| `test_hsv_segmenter.py` | HSV segmenter (synthetics) |
| `test_pose_geometry.py` | geometry + geometric (&lt;5 mm, yaw &lt;2°, 2 mm noise) |
| `test_yolo_seg.py` | slots / `verify_hsv` (no ultralytics) |
| `test_autolabel_helpers.py` | bag split, YOLO polygon (no GPU) |
