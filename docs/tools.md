# Tools (`tools/`)

Recording, frame extraction, labeling helpers, YOLO train/export/bench, and robot deploy scripts. Run from `cwd=/ws` after `source install/setup.bash`.

## `record_cubes.sh` — record a bag

Requires a live RealSense camera.

```bash
./tools/record_cubes.sh <layout> <light> [duration_s] [notes] [yaw_deg]
```

| Argument | Description |
|----------|-------------|
| `layout` | `R` / `G` / `X` from left (e.g. `RGGR`) |
| `light` | e.g. `dzienne`, `mocne` |
| `duration_s` | seconds (default `15`) |
| `notes` | free text |
| `yaw_deg` | angles in degrees, comma-separated; empty field for `X` |

```bash
./tools/record_cubes.sh RGGR mocne 15 "" "0,15,45,30"
./tools/record_cubes.sh RXGR mocne 15 "" "0,,45,30"
```

Writes: `/bags/kostki_.../` + a row in `/bags/labels.csv` (first run also adds the `yaw_deg` column).

## `record_balls.sh` — record a balls bag (stage 2)

Requires a live RealSense camera. Same topics as `record_cubes.sh`.

```bash
./tools/record_balls.sh <ball_count> <shelf> <light> [duration_s] [notes]
```

| Argument | Description |
|----------|-------------|
| `ball_count` | `0`–`4` |
| `shelf` | `h60`, `h100`, `h60_100` (both shelves), or `stol` |
| `light` | e.g. `dzienne`, `lampa` (letters, digits, `_`) |
| `duration_s` | seconds (default `10`) |
| `notes` | free text |

```bash
./tools/record_balls.sh 4 h60 lampa
./tools/record_balls.sh 2 h60_100 dzienne 15 "jedna przy krawedzi"
```

Writes: `/bags/pilki_<count>_<shelf>_<light>_<date>/` + a row in `/bags/labels_pilki.csv` (`bag, liczba_pilek, polka, swiatlo, czas_s, uwagi`). Bad args = exit with no recording and no label. Frames: `python3 tools/extract_frames.py --labels /bags/labels_pilki.csv --out-dir data_pilki` (`index.csv` also gets `liczba_pilek`, `polka`).

## `labels_csv.py` — CSV helper

```bash
python3 tools/labels_csv.py ensure /bags/labels.csv
python3 tools/labels_csv.py append /bags/labels.csv \
  --bag NAME --ulozenie RGGR --swiatlo dzienne --czas-s 15 \
  --uwagi "" --yaw-deg "0,15,45,30"
python3 tools/labels_csv.py ensure-balls /bags/labels_pilki.csv
python3 tools/labels_csv.py append-balls /bags/labels_pilki.csv \
  --bag NAME --liczba-pilek 3 --polka h60 --swiatlo lampa --czas-s 10 --uwagi=""
```

## `extract_frames.py` — frames from MCAP

Uses the **rosbags** library (no `rclpy`). Color↔depth sync ≤ 20 ms.

```bash
python3 tools/extract_frames.py
python3 tools/extract_frames.py --every 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--every` | `10` | keep every Nth color frame |
| `--max-dt-ms` | `20` | max color↔depth Δt |
| `--out-dir` | `data` | output root |

```text
data/frames/<bag>/<bag>_<n>_color.png
data/frames/<bag>/<bag>_<n>_depth.npy  # float32 [m], 0 = missing
data/frames/<bag>/camera_info.json
data/index.csv                        # + ulozenie, swiatlo, yaw_deg
```

## `apriltag_from_bag.py` — AprilTag poses from bags

Uses **rosbags** (no `rclpy`; MCAP and sqlite3 bags). For each frame (every `--every`) and each tag with a pose, writes a row in the camera frame plus one row per `--frames` frame, transformed with `/tf` + `/tf_static` from the bag (nearest TF sample on each edge, no interpolation, max `--max-tf-dt-ms`).

```bash
python3 tools/apriltag_from_bag.py /bags/tagi_* --tag-size 0.162 --frames pelvis,left_ankle_roll_link
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tag-size` | `0.10` | black square side [m] |
| `--family` / `--ids` | `36h11` / all | family, id filter (`0,1,2`) |
| `--frames` | — | extra TF frames |
| `--every` | `1` | every Nth frame |
| `--max-reproj-px` | `2.0` | same as the node |
| `--max-tf-dt-ms` | `50` | max TF sample distance from image stamp |
| `--out` | `data/apriltags.csv` | output |

Columns: `bag, stamp_s, tag_id, frame, x, y, z, qx, qy, qz, qw, distance_m, reproj_px, ambiguity` (`ambiguity` near 1 = tag could be tilted the other way, pose uncertain; small / distant tag).

## `roi_picker.py` — cube slots

```bash
python3 tools/roi_picker.py data/frames/<bag>/<bag>_0_color.png
```

Click order = `slot` 0,1,2,… → `src/g1_perception/config/rois.yaml`.

## `hsv_tuner.py` — HSV thresholds

```bash
python3 tools/hsv_tuner.py
```

| Key | Action |
|-----|--------|
| `n` / `p` | next / previous frame |
| `s` | save `hsv.yaml` |
| `q` / `ESC` | quit |

## `autolabel.py` — Grounded SAM → YOLO-seg

Ontology `{"cube": "kostka"}`. Class color from ROI + `ulozenie` (R→`czerwona`=0, G→`zielona`=1). Split **by bag**.

```bash
# Colab GPU recommended
pip install autodistill autodistill-grounded-sam supervision
python3 tools/autolabel.py --index data/index.csv --visualize
```

Output: `datasets/cubes_seg/{images,labels}/{train,val}/`, `data.yaml`, optional `visualizations/`.

## `train_yolo.py` — Ultralytics training

`ultralytics` is in the image (with `numpy<2`; see [environment.md](environment.md)).

```bash
python3 tools/train_yolo.py                       # seg: datasets/cubes_seg, yolo26n-seg.pt
python3 tools/train_yolo.py --task pose --device 0  # pose: datasets/cubes_pose, yolo26n-pose.pt
python3 tools/train_yolo.py --task pose --model yolo26s-pose.pt --epochs 150
```

`hsv_h=0` (color is the class); for pose `fliplr=0`; dataset paths are resolved to absolute. Weights: `runs/seg/cubes/weights/best.pt`, `runs/pose/cubes_pose/weights/best.pt`.

Full data/model flow (CVAT → datasets → train → export → robot): [training.md](training.md).

Also see (not expanded here): `cvat_to_yolo.py`, `preview_pose_labels.py`, `export_model.py`, `bench_models.py`, `compare_backends.py`, `jetson_gpu_check.py`, `deploy_robot.sh`, `webcam_yolo.py`.
