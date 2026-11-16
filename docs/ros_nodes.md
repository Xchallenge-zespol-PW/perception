# ROS 2 nodes

`perception_node` runs the cube pipeline on synced RGB-D + `camera_info` and publishes detections / TF for exercise 1. `apriltag_node` is a separate AprilTag detector.

Parameter tables and YAML: [config.md](config.md). Robot deploy: [robot.md](robot.md).

## `perception_node`

Camera ~15 fps; CPU inference ~125 ms (~8 fps): the sync callback stores **only the latest** set (color + depth + `camera_info`, sensor-data QoS). A worker thread takes that set, builds a `Frame` (rgb8→RGB; YOLO converts to BGR internally; 16UC1 mm→float32 m), then runs `Pipeline.process` + `SlotVoter`. Frames that arrive during inference are **dropped** (no queue = no growing lag). Publishing `/perception/cubes`, masks, and debug uses timer `publish_rate_hz`. The model is loaded and warmed up at node start (with CPU fallback). Every 10 s the log reports mean inference [ms], skipped frames, and active backend. Before Ultralytics, `YOLO_AUTOINSTALL=false` is set.

### Launch on a bag

Terminal 1 — loop playback:

```bash
source /opt/ros/humble/setup.bash
ros2 bag play /bags/kostki_RGGR_niskie_20261015_184542 --loop
```

Terminal 2 — node (`cwd=/ws` because `model_path` is relative):

```bash
cd /ws
source /opt/ros/humble/setup.bash && source install/setup.bash
# laptop (default config:=params_laptop.yaml):
ros2 launch g1_perception perception.launch.py
# robot:
ros2 launch g1_perception perception.launch.py config:=params_robot.yaml
# override ROI without editing YAML:
ros2 run g1_perception perception_node --ros-args \
  --params-file install/g1_perception/share/g1_perception/config/params_laptop.yaml \
  -p rois_path:=src/g1_perception/config/rois_RGGR_niskie_184542.yaml
```

Terminal 3 — inspect:

```bash
ros2 topic echo /perception/cubes
ros2 run rqt_image_view rqt_image_view /perception/debug_image
```

Notes:

- `--loop` rewinds stamps — keep `use_sim_time:=false`. Use `use_sim_time:=true` only with `ros2 bag play --clock` and without `--loop`.
- `rois.yaml` / `rois_path` must match the frame. Pixel ROIs need a fixed camera–table pose.
- `config/*.yaml` install into `share/g1_perception/config/` (`setup.py` glob).
- On the robot, TensorRT on GPU is default; ONNX CPU is the fallback. `skipped_frames` in the 10 s log shows how many 15 Hz frames were dropped.

```bash
ros2 launch g1_perception perception.launch.py config:=params_laptop.yaml
# robot, force CPU without editing YAML:
ros2 launch g1_perception perception.launch.py config:=params_robot.yaml \
  model_path:=/ws/models/cubes_seg.onnx device:=cpu
```

### Message contract

**`/perception/cubes`** — `vision_msgs/Detection3DArray`:

- **`rois_mode=image_roi`**: one `Detection3D` per ROI, slot order (`id` = `"0"`…).
- **`rois_mode=none`**: one `Detection3D` per tracked cube. After successful validation `id` = slot number `"1"`…`"N"` (sort along `slot_axis` in `target_frame`). If validation fails (e.g. cubes not in a row): `id` = `"t<tracker id>"`, voted color/pose kept, warning in the log.

| Field | Value |
|-------|-------|
| `header.stamp` | stamp of the processed color image |
| `header.frame_id` | `target_frame` (tracker already works in this frame because the camera moves with the torso); camera frame when `target_frame` is empty. With `rois_mode=none` and missing TF the frame is not tracked: empty list in camera frame, warning in the log |
| `detections[i].header` | same as `header` |
| `detections[i].id` | slot / place number or `t<track id>` (when assignment failed) |
| `detections[i].results` | exactly 1 element |
| `results[0].hypothesis.class_id` | `red` \| `green` \| `empty` \| `uncertain` |
| `results[0].hypothesis.score` | share of window frames matching the result, [0, 1] |
| `results[0].pose.pose` | cube pose (center, xyzw quaternion; +Z = table normal, yaw mod 90°) |
| `results[0].pose.covariance` | zeros (unused) |
| `bbox.center` | same as `results[0].pose.pose` |
| `bbox.size` | `cube_size_m` on each axis when pose exists; `0` otherwise |

Pose is only for `red` / `green`. For `empty`, `uncertain`, or color without a successful pose: `bbox.size = 0` and pose fields are zero (identity orientation) — ignore them.

**`/perception/debug_image`** — `sensor_msgs/Image` `rgb8`, camera frame: ROIs (if any), masks, `id:label status` (+ yaw), inference time in ms. Published only when someone subscribes.

**`/perception/mask`** — `sensor_msgs/Image` `mono8`, camera frame: sum of detection masks in the current frame (0 / 255).

### Exercise 1 (`cube_sorting_node` from `unitree_controllers`)

Only with `rois_mode=none`. Controller logic reads all frames relative to `pelvis` and computes the table from `<cube>_start` frames and feet.

**Freeze start layout** (`core.cube_registry`): when for `latch_s` seconds of frame stamps (so independent of GPU/CPU) slot assignment is valid, every cube has voted color `red`/`green`, and tracks / slots / colors do not change, place `n` becomes cube `<cube_name_prefix>n` (`cube_1`…`cube_4`). The layout is published once:

| Output | Content |
|--------|---------|
| `/tf_static` | `<start_frame> → <cube>_start` for all cubes in one message (`StaticTransformBroadcaster` in rclpy keeps only the last); pose from voting. Sent before colors |
| `cube_colors_topic` (`/cube_colors`) | `xchallenge_msgs/CubeColors`, reliable + transient local, depth 1: `cube_names[i]` has color `colors[i]`, in place order. One message per layout; controller consumes it for one attempt |

Start frames hang under `start_frame` because the controller re-reads start on every line check: on the robot that is the foot (stays put while the pelvis sways; no odometry dependency). Without TF `start_frame ← target_frame` the layout is not frozen (warning in the log).

**Live poses**: `/tf` `<target_frame> → <cube>` at `publish_rate_hz` for each layout cube, stamped with current time; a cube off-camera keeps its last seen pose. Additionally `<target_frame> → <cube>_seen` only for cubes seen in that frame, stamped with the frame: its age shows how fresh the pose is (`cube_sorting_node` reads these, `cube_seen_suffix` in `task_1.yaml`). The name sticks to its track even when the cube leaves the line. When the track disappears (hand occludes, camera rotates with the waist) or votes the other color, the name is taken by a new track within `reid_gate_m` of the last pose along the cube line: push is across the column and the neighbour is 6 cm away. The line is the start layout’s main axis (`slot_axis` only for layouts with spread under 2 cm), because with the robot yawed 5° to the table a cube pushed 30 cm moves 2.6 cm along `slot_axis`. Log: `cube_N found again as track …`. When the stamp jumps backward (bag `--loop`), the tracker renumbers tracks, so each name forgets its track and is re-found by position.

**Pose update gates** (`rois_mode=none` only): when a cube is clipped or occluded, or the camera rotates quickly, that frame does not update the cube pose. The track still lives (no drop, no new id) and votes color, but the position does not enter the median, so `<cube>` keeps the last good pose. Such a frame also does not spawn a new track. Counters in `perception stats`: `hidden_poses` (cubes), `turning_frames` (frames).

| Param | On robot | Meaning |
|-------|----------|---------|
| `pose_border_px` | 3 | mask closer to the image border than this many px = cube clipped |
| `min_visible_ratio` | 0.45 | mask area / cube face at that distance. Full cubes from 11.2026 bags (model v4, 1529 dets): 0.47–1.7, p1 0.61, median 0.91 (view-angle dependent); 0.45 catches cubes roughly half-occluded or worse |
| `max_camera_rot_deg_s` | 20 | camera rotation in `target_frame` between successive frames (from TF). Tune on bags with `/tf` (`record_cubes.sh` records it since 11.2026) |

`0` disables that gate.

**`~/reset`** (`std_srvs/Trigger`, `/perception_node/reset`): forgets layout and tracks. Call before each new attempt (colors are randomized each time); the new layout gets new start frames and a new colors message.

Requirement: camera must be in the robot TF tree (`pelvis` → camera optical frame); otherwise frames are not tracked (`/perception/cubes` empty, debug image shows raw detections) and nothing is published for exercise 1.

---

## `apriltag_node`

Separate node, independent of cubes: detects AprilTags in the color image (OpenCV aruco, families `16h5` / `25h9` / `36h10` / `36h11`; works on OpenCV 4.5 contrib and ≥ 4.7) and estimates each tag’s pose from its 4 corners (`solvePnP` IPPE, no depth). Detection ~1–2 ms per 640×480 frame (laptop).

```bash
ros2 launch g1_perception apriltag.launch.py                       # config/apriltag.yaml
ros2 launch g1_perception apriltag.launch.py tag_size_m:=0.162 target_frame:=pelvis
```

| Output | What |
|--------|------|
| TF `tag_<id>` | under camera optical frame, image stamp. Tag center, x right, y up, z out of the printed face (for a floor tag: up) |
| `/apriltag/detections` (`vision_msgs/Detection3DArray`) | `id` = tag number, `class_id` = `tag36h11`, `score` = 1 / (1 + reprojection error px), pose in `target_frame` (when its TF is known, else camera frame), `bbox.size` = tag side |
| `/apriltag/summary` (`std_msgs/String`) | per frame JSON of id and camera–tag-center distance: `[{"id": 7, "distance_m": 1.832}]` (`[]` = no tags with pose); distance always from camera, independent of `target_frame` |
| `/apriltag/debug_image` | outlines, ids, axes (x red, y green, z blue); drawn only when subscribed |
| log every `log_period_s` | `tags: 7: 1.83 m, 3: 2.41 m` (only when something is visible; `0` = off) |
| log every `stats_period_s` | `frames`, `with_tags`, `ids`, `detect_ms_avg` |

Params (`config/apriltag.yaml`): `family`, `tag_size_m` (side of the **black** square, not the white border), `tag_ids` (`"0,1,2"`, `""` = all), `max_reproj_px` (worse corner fit = no pose), `target_frame`, `tf_prefix`, `publish_tf`, `image_topic`, `camera_info_topic`, `summary_topic`, `log_period_s`.

Inspect ids and distances: `ros2 topic echo /apriltag/summary`. Distance scales with `tag_size_m` (wrong side length scales distance by the same factor).

Recording for later offline analysis (`tools/apriltag_from_bag.py`) must include TF, otherwise tags cannot be transformed into robot frames:

```bash
ros2 bag record -s mcap -o tagi_$(date +%Y%m%d_%H%M%S) \
  /camera/camera/color/image_raw /camera/camera/color/camera_info /tf /tf_static
```

Tags: matte print, flat, with a white border; measure the black square side; unique ids. Walking motion blur ruins corners — brighter light / shorter camera exposure helps.
