# Offline workflow

Record RealSense bags → extract frames → label / autolabel → train YOLO → run the ROS-free pipeline or `perception_node`. Rosbags stay on `/bags/` (not in git).

```text
1. Record bag                 ./tools/record_cubes.sh ...   (balls: record_balls.sh)
2. Extract frames             python3 tools/extract_frames.py
3. Mark ROIs                  python3 tools/roi_picker.py <png>
4. Tune HSV                   python3 tools/hsv_tuner.py
5a. Baseline HSV+geometric    Pipeline.from_config(hsv / geometric)
5b. Or YOLO dataset           python3 tools/autolabel.py --visualize
6. Train YOLO-seg             python3 tools/train_yolo.py
7. YOLO inference             Pipeline.from_config(yolo_seg / geometric)
```

Full data/model flow (CVAT → datasets → train → export → robot): [training.md](training.md). Tool reference: [tools.md](tools.md).

## Data and topics

**`/bags/labels.csv`:** `bag`, `ulozenie`, `swiatlo`, `czas_s`, `uwagi`, `yaw_deg`

**Topics:**

- `/camera/camera/color/image_raw` (`rgb8`)
- `/camera/camera/aligned_depth_to_color/image_raw` (`16UC1`, mm)
- `/camera/camera/color/camera_info`
- `/tf`, `/tf_static` (`record_cubes.sh`; `/tf` since 11.2026: camera motion in the bag, e.g. for tuning `max_camera_rot_deg_s`)

**Layout code:** `R`/`G`/`X` from left (e.g. `RGGR`).

Training frames land under `data/` (gitignored). Datasets under `datasets/`, Ultralytics runs under `runs/`. Competition / lab bags are a local data source — do not commit MCAP files; see [environment.md](environment.md#artifacts-not-in-git).
