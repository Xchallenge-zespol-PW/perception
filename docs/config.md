# Configuration (YAML and node params)

Laptop vs robot params live in `params_*.yaml`. ROI / HSV / label maps are separate YAML files under `src/g1_perception/config/`.

## Node params (`config/params_*.yaml`)

Same keys: `params_laptop.yaml` (`.pt` / `.onnx`, `device: cpu`), `params_robot.yaml` (`.engine`, `device: "0"`, fallback `.onnx` on CPU). G1/PC2 deploy: [robot.md](robot.md), `deploy/requirements-robot.txt`.

| Group | Parameters |
|-------|------------|
| Input | `color_topic`, `depth_topic` (aligned, `16UC1` mm or `32FC1` m), `camera_info_topic`, `sync_queue_size`, `sync_slop_s` |
| Output | `output_topic` (`/perception/cubes`), `debug_image_topic`, `mask_topic`, `target_frame` (`""` = camera frame), `tf_timeout_s` |
| Pipeline | `segmenter` (`yolo_seg` \| `hsv`), `pose_estimator` (`geometric` \| `centroid` \| `pnp`), `model_path`, `device`, `fallback_model_path` / `fallback_device` (when `model_path` fails on `device`), `warmup_runs` (empty-image preds at start), `conf`, `verify_hsv`, `rois_mode` (`none` \| `image_roi`), `rois_path` / `hsv_path` / `labels_path` (`""` = defaults; in `none` a missing ROI file is not an error), `duplicate_overlap` (in `none`: a mask that overlaps a higher-scoring detection by more than this fraction of the smaller mask is treated as the same cube seen twice — e.g. both colors, or full cube vs top face — and dropped regardless of class; 0 disables) |
| Pose | `cube_size_m`, `margin_px` (in `none` with `geometric`: one table plane per frame, with `margin_px` around each detection without using all detection masks; cube height is plane + `cube_size_m`/2, so a per-cube plane without neighbour masks used to place the middle cube of a column on neighbours’ top faces, 5 cm too high), `z_min_m`, `z_max_m` (≤ 0 disables); shape (`geometric`): `min_height_m` (only mask points from this height up to 1.5 × `cube_size_m` above the table enter the pose; ≤ 0 disables), `min_above_ratio` (too few points above the table = flat clutter, e.g. tape, rejected), `footprint_min_m` / `footprint_max_m` (table rectangle; 0 disables) |
| PnP (`pose_estimator: pnp`) | `kpt_min_conf`, `pnp_max_reproj_px`, `pnp_max_depth_diff_m` (with `pnp_depth_diff_ref_m`, then × (z / ref)²), `pnp_max_normal_deg`, `pnp_fallback` (`geometric` \| `none`) |
| Voting / tracking | `vote_window_s` (> 0 = time window in frame-stamp seconds; 0 = frame window), `vote_min_votes` (min votes in time mode), `vote_window` (frame window; in time mode caps votes), `vote_min_ratio`, `vote_min_frames` (0 = `vote_window`), `match_dist_m` (in 3D, or with `match_max_dz_m` > 0 in x,y with height delta up to `match_max_dz_m`: a bad height in one frame does not spawn a new track, and the median ignores it; a detection that close to a track already claimed or created in this frame is a second mask of the same cube and does not spawn a track), `max_missed_s` (> 0 = track dies after that many seconds without detection) / `max_missed` (frames), `fresh_s` (> 0 = only tracks seen in the last this many seconds count for slots and freeze; a drifted track does not turn one cube into two) |
| Slots (`rois_mode=none`) | `slot_axis` (numbering: place 1 = smallest coordinate; on robot `x`, `cube_1` nearest), `expected_count`, `max_line_deviation_m`, `min_spacing_m` / `max_spacing_m` (neighbour spacing along the fitted row line, so the row may be angled vs `slot_axis`); with more tracks (up to 8) passes when exactly one `expected_count`-subset satisfies these and extras are ignored |
| Exercise 1 (`rois_mode=none`) | `cube_colors_topic`, `cube_name_prefix`, `start_frame` (`""` = `target_frame`), `latch_s` (how many stamp-seconds the layout must stay stable before freeze), `reid_gate_m` |
| Rate | `publish_rate_hz` |

`params_robot.yaml` uses `rois_mode: none` (YOLO → 3D tracker → TF → place numbers) and exercise-1 settings: `target_frame: left_ankle_roll_link` (foot stays put; pelvis shifts ~2 cm on waist yaw and sways on push), `slot_axis: x` (cube column along the robot view, centers every 6 cm, `cube_1` nearest), `start_frame: left_ankle_roll_link`. `params_laptop.yaml` stays in `camera_link` with `slot_axis: y`.

## Calibration YAML

`src/g1_perception/config/` (local calibration — usually do not commit ROI/HSV):

**`rois.yaml`**

```yaml
rois:
  - slot: 0
    x0: 100
    y0: 200
    x1: 180
    y1: 280
```

**`hsv.yaml`**

```yaml
red_low:  {h_min: 0,   h_max: 10,  s_min: 80, s_max: 255, v_min: 50, v_max: 255}
red_high: {h_min: 170, h_max: 180, s_min: 80, s_max: 255, v_min: 50, v_max: 255}
green:    {h_min: 40,  h_max: 90,  s_min: 40, s_max: 255, v_min: 40, v_max: 255}
```

**`labels.yaml`** (in repo) — sole place mapping model class names to canonical labels. Keys are case-insensitive; values only `red` / `green` / `empty` / `uncertain`. A new model with different class names needs a new entry here.

```yaml
label_map:
  red cube: red
  czerwona: red
  green cube: green
  zielony: green
  brak: empty
  nieznany: uncertain
```
