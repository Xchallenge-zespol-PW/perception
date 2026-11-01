# g1_perception

Percepcja kostek dla robota **Unitree G1 EDU**: na 4 znanych miejscach stołu wykryć kostkę (czerwona / zielona / brak, sześcian ~5 cm) i wyznaczyć jej pozę (pozycja 3D + orientacja).

Środowisko: **ROS 2 Humble**, Python 3.10, `numpy<2` (kompatybilność z `cv_bridge`), praca w **devcontainerze**. Logika percepcji jest niezależna od ROS; węzeł ROS 2 będzie tylko cienką warstwą na końcu.

## Architektura

```text
Frame (RGB + depth + K)
        │
        ▼
   Segmenter  →  list[Detection]     # "hsv" | "yolo_seg"
        │
        ▼
 PoseEstimator  →  Pose | None       # "centroid" | "geometric"
        │
        ▼
 list[SlotResult]                    # dokładnie 1 na ROI: red | green | empty | uncertain
```

| Warstwa | Co jest |
|---------|---------|
| `core.types` | `Frame`, `Detection`, `Pose`, `SlotResult`, stałe etykiet, protokoły |
| `core.config` | ROI / HSV / mapa etykiet z YAML |
| `core.labels` | `canonical_label` (nazwa z modelu → etykieta kanoniczna) |
| `core.voting` | `SlotVoter`: głosowanie po oknie N klatek per slot |
| `core.dataset` | `load_frame` z `data/index.csv` |
| `core.geometry` | deprojekcja, RANSAC płaszczyzny, poza kostki, `transform_pose` |
| `core.frame_utils` | głębia mm → m, `K` z `camera_info` |
| `core.debug_vis` | obraz debug (ROI, maski, etykiety, yaw), suma masek mono8 |
| `pipeline` | `@register`, `Pipeline.from_config` / `process` |
| `segmenters/` | `hsv`, `yolo_seg` |
| `pose/` | `centroid`, `geometric` |
| `node_config` | parametry węzła → config `Pipeline` / `SlotVoter` (bez ROS) |
| `ros_conversions` | `VotedSlot` → `vision_msgs/Detection3DArray` |
| `perception_node` | cienki węzeł rclpy (sync, cv_bridge, TF, publikacja) |
| `tools/` | nagranie → klatki → kalibracja → autolabel → trenowanie |

Logika (`core/`, `segmenters/`, `pose/`, `pipeline`, `node_config`) nie importuje ROS. Planowane: `evaluate`, FoundationPose, eksport TensorRT (`.engine` już obsługiwany w loaderze).

## Struktura repo

```text
/ws
├── docker/Dockerfile              # ROS Humble + RealSense + OpenCV + rosbags + ultralytics
├── docker/constraints.txt         # PIP_CONSTRAINT: numpy<2, opencv-python<4.12, setuptools<80
├── .devcontainer/                 # /ws, /bags, sieć host, udev
├── src/g1_perception/             # pakiet ament_python
│   ├── config/                    # rois.yaml, hsv.yaml (narzędzia), labels.yaml, params.yaml
│   ├── launch/perception.launch.py
│   ├── g1_perception/
│   │   ├── core/
│   │   │   ├── types.py
│   │   │   ├── dataset.py
│   │   │   ├── config.py
│   │   │   ├── labels.py
│   │   │   ├── voting.py
│   │   │   ├── frame_utils.py
│   │   │   ├── debug_vis.py
│   │   │   └── geometry.py
│   │   ├── node_config.py
│   │   ├── ros_conversions.py
│   │   ├── perception_node.py     # entry point: perception_node
│   │   ├── segmenters/
│   │   │   ├── hsv.py             # @register("hsv")
│   │   │   └── yolo_seg.py        # @register("yolo_seg")
│   │   ├── pose/
│   │   │   ├── centroid.py        # @register("centroid")
│   │   │   └── geometric.py       # @register("geometric")
│   │   └── pipeline.py
│   └── test/
├── tools/
│   ├── record_cubes.sh
│   ├── labels_csv.py
│   ├── extract_frames.py
│   ├── roi_picker.py
│   ├── hsv_tuner.py
│   ├── autolabel.py               # Grounded SAM → YOLO-seg dataset
│   └── train_yolo.py              # Ultralytics yolo11n-seg
├── data/                          # klatki + index.csv (gitignore)
├── datasets/                      # cubes_seg / (gitignore)
├── runs/                          # trening YOLO (gitignore)
└── results/                       # ewaluacja (gitignore)
```

Poza repo (bind mount): `/bags/` — MCAP + `labels.csv`.

## Uruchomienie środowiska

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select g1_perception
source install/setup.bash
```

### NumPy < 2 (wymóg `cv_bridge`)

`cv_bridge` z ROS 2 Humble (apt) jest skompilowany pod NumPy 1.x. Z NumPy 2 import kończy się błędem `AttributeError: _ARRAY_API not found` („A module that was compiled using NumPy 1.x cannot be run in NumPy 2…”), więc węzeł ROS nie wystartuje. Typowa przyczyna: `pip install ultralytics` (np. `--user`) bez ograniczeń dociąga najnowszy `numpy` 2.x i `opencv-python` ≥ 4.12 (które samo wymaga NumPy 2); pakiety w `~/.local` przesłaniają wtedy wersje z obrazu.

Zabezpieczenie w obrazie:

- `docker/constraints.txt` (`numpy<2`, `opencv-python<4.12`, `opencv-python-headless<4.12`) jest kopiowany do `/etc/pip/constraints.txt` i ustawiony jako `PIP_CONSTRAINT`, więc obowiązuje każde `pip install` w kontenerze, także `--user`.
- `ultralytics` jest instalowany w obrazie już z tymi ograniczeniami.
- Build kończy się krokiem weryfikacji (failuje, jeśli NumPy ≥ 2 albo `cv_bridge` się nie importuje):

```bash
source /opt/ros/humble/setup.bash
python3 -c "import numpy, cv2; from cv_bridge import CvBridge; assert numpy.__version__.startswith('1')"
```

Naprawa działającego kontenera bez przebudowy:

```bash
pip3 install --user -c docker/constraints.txt "numpy<2" "opencv-python<4.12" "opencv-python-headless<4.12"
```

Ten sam plik pinuje `setuptools<80`: od wersji 80 nie ma `setup.py develop`, więc `colcon build --symlink-install` kończy się `error: option --uninstall not recognized`. Naprawa: `pip3 install --user -c docker/constraints.txt "setuptools<80"`.

### Testy

```bash
# zalecane w tym kontenerze (konflikt anyio / systemowy pytest 6.2)
cd /ws/src/g1_perception && python3 -m pytest -p no:anyio -q && cd /ws

colcon test --packages-select g1_perception
colcon test-result --verbose
```

Nie instaluj `pytest>=7` przez `pip --user` — gryzie się z pluginami ROS (`launch_testing`).

---

## Węzeł ROS 2 (`perception_node`)

Synchronizuje kolor + głębię aligned + `camera_info` (`ApproximateTimeSynchronizer`), buduje `Frame` (RGB, głębia w metrach), uruchamia `Pipeline.process` i `SlotVoter`, publikuje wynik w `target_frame`. Przetwarzana jest najnowsza zsynchronizowana klatka, z częstotliwością `publish_rate_hz` (starsze klatki są pomijane).

### Uruchomienie na bagu

Terminal 1 — odtwarzanie w pętli:

```bash
source /opt/ros/humble/setup.bash
ros2 bag play /bags/kostki_GRGR_mocne_20261015_184734 --loop
```

Terminal 2 — węzeł (z `cwd=/ws`, bo `model_path` w `params.yaml` jest względny):

```bash
cd /ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch g1_perception perception.launch.py
# własne parametry (np. inne ROI):
ros2 launch g1_perception perception.launch.py params_file:=/ścieżka/params.yaml
# bez launch, z nadpisaniem pojedynczego parametru:
ros2 run g1_perception perception_node --ros-args \
  --params-file install/g1_perception/share/g1_perception/config/params.yaml \
  -p rois_path:=/ścieżka/rois.yaml
```

Terminal 3 — podgląd:

```bash
ros2 topic echo /perception/cubes
ros2 run rqt_image_view rqt_image_view /perception/debug_image
```

Uwagi:

- `--loop` cofa znaczniki czasu na początku każdej pętli — zostaw `use_sim_time:=false` (domyślnie). `use_sim_time:=true` tylko z `ros2 bag play --clock` bez `--loop`.
- `rois.yaml` musi pasować do kadru nagrania. ROI są w pikselach, więc wymagają stałej pozycji kamery względem stołu.
- `params.yaml` jest instalowany do `share/g1_perception/config/`; po zmianie w `src/` z `--symlink-install` działa od razu.
- Bez GPU (devcontainer na CPU) YOLO daje ~2–5 Hz; `publish_rate_hz` to górny limit.

### Parametry (`config/params.yaml`)

| Grupa | Parametry |
|-------|-----------|
| Wejście | `color_topic`, `depth_topic` (aligned, `16UC1` mm lub `32FC1` m), `camera_info_topic`, `sync_queue_size`, `sync_slop_s` |
| Wyjście | `output_topic` (`/perception/cubes`), `debug_image_topic`, `mask_topic`, `target_frame` (`""` = ramka kamery), `tf_timeout_s` |
| Pipeline | `segmenter` (`yolo_seg` \| `hsv`), `pose_estimator` (`geometric` \| `centroid`), `model_path`, `device`, `conf`, `verify_hsv`, `rois_path` / `hsv_path` / `labels_path` (`""` = domyślne z `config/`) |
| Poza | `cube_size_m`, `margin_px`, `z_min_m`, `z_max_m` (≤ 0 wyłącza) |
| Głosowanie | `vote_window`, `vote_min_ratio`, `vote_min_frames` (0 = `vote_window`) |
| Tempo | `publish_rate_hz` |

### Kontrakt wiadomości

**`/perception/cubes`** — `vision_msgs/Detection3DArray`, jedna `Detection3D` na każde ROI z `rois.yaml`, w kolejności slotów:

| Pole | Wartość |
|------|---------|
| `header.stamp` | stempel obrazu kolorowego przetworzonej klatki |
| `header.frame_id` | `target_frame`; ramka kamery (`camera_color_optical_frame`), gdy `target_frame` puste albo brak TF (wtedy warning w logu) |
| `detections[i].header` | jak `header` |
| `detections[i].id` | numer slotu jako string (`"0"`…`"3"`) |
| `detections[i].results` | dokładnie 1 element |
| `results[0].hypothesis.class_id` | `red` \| `green` \| `empty` \| `uncertain` (wynik głosowania) |
| `results[0].hypothesis.score` | udział klatek w oknie zgodnych z wynikiem, [0, 1] |
| `results[0].pose.pose` | poza kostki (środek, kwaternion xyzw; +Z = normalna stołu, yaw modulo 90°) |
| `results[0].pose.covariance` | zera (nieużywane) |
| `bbox.center` | jak `results[0].pose.pose` |
| `bbox.size` | `cube_size_m` na każdej osi, gdy jest poza; `0`, gdy jej brak |

Poza jest tylko dla `red` / `green`. Dla `empty`, `uncertain` albo koloru bez udanej estymacji pozy: `bbox.size = 0`, a pola pozy są zerowe (orientacja jednostkowa) i należy je ignorować.

**`/perception/debug_image`** — `sensor_msgs/Image` `rgb8`, ramka kamery: ROI w kolorze wyniku głosowania, maski bieżącej klatki, opis `slot:etykieta score yaw`. Publikowany tylko, gdy ktoś subskrybuje.

**`/perception/mask`** — `sensor_msgs/Image` `mono8`, ramka kamery: suma masek detekcji przypisanych do slotów w bieżącej klatce (0 / 255), wejście dla FoundationPose.

---

## Typowy workflow offline

```text
1. Nagraj bag                 ./tools/record_cubes.sh ...
2. Wyciągnij klatki           python3 tools/extract_frames.py
3. Zaznacz ROI                python3 tools/roi_picker.py <png>
4. Ustaw HSV                  python3 tools/hsv_tuner.py
5a. Baseline HSV+geometric    Pipeline.from_config(hsv / geometric)
5b. Albo dataset YOLO         python3 tools/autolabel.py --visualize
6. Trenuj YOLO-seg            python3 tools/train_yolo.py
7. Inference YOLO             Pipeline.from_config(yolo_seg / geometric)
```

---

## Narzędzia (`tools/`)

Komendy z `cwd=/ws` i `source install/setup.bash`.

### `record_cubes.sh` — nagrywanie baga

Wymaga działającej kamery RealSense.

```bash
./tools/record_cubes.sh <ulozenie> <swiatlo> [czas_s] [uwagi] [yaw_deg]
```

| Argument | Opis |
|----------|------|
| `ulozenie` | `R` / `G` / `X` od lewej (np. `RGGR`) |
| `swiatlo` | np. `dzienne`, `mocne` |
| `czas_s` | sekundy (domyślnie `15`) |
| `uwagi` | tekst |
| `yaw_deg` | kąty w stopniach po przecinku; dla `X` puste pole |

```bash
./tools/record_cubes.sh RGGR mocne 15 "" "0,15,45,30"
./tools/record_cubes.sh RXGR mocne 15 "" "0,,45,30"
```

Zapis: `/bags/kostki_.../` + wiersz w `/bags/labels.csv` (przy pierwszym uruchomieniu dopisze kolumnę `yaw_deg`).

### `labels_csv.py` — helper CSV

```bash
python3 tools/labels_csv.py ensure /bags/labels.csv
python3 tools/labels_csv.py append /bags/labels.csv \
  --bag NAME --ulozenie RGGR --swiatlo dzienne --czas-s 15 \
  --uwagi "" --yaw-deg "0,15,45,30"
```

### `extract_frames.py` — klatki z MCAP

Biblioteka **rosbags** (bez `rclpy`). Sync kolor↔głębia ≤ 20 ms.

```bash
python3 tools/extract_frames.py
python3 tools/extract_frames.py --every 5
```

| Flaga | Domyślnie | Opis |
|-------|-----------|------|
| `--every` | `10` | co którą klatkę kolorową |
| `--max-dt-ms` | `20` | max Δt kolor↔głębia |
| `--out-dir` | `data` | wyjście |

```text
data/frames/<bag>/<nr>_color.png
data/frames/<bag>/<nr>_depth.npy      # float32 [m], 0 = brak
data/frames/<bag>/camera_info.json
data/index.csv                        # + ulozenie, swiatlo, yaw_deg
```

### `roi_picker.py` — miejsca kostek

```bash
python3 tools/roi_picker.py data/frames/<bag>/0_color.png
```

Kolejność zaznaczeń = `slot` 0,1,2,… → `src/g1_perception/config/rois.yaml`.

### `hsv_tuner.py` — progi HSV

```bash
python3 tools/hsv_tuner.py
```

| Klawisz | Akcja |
|---------|--------|
| `n` / `p` | następna / poprzednia klatka |
| `s` | zapisz `hsv.yaml` |
| `q` / `ESC` | wyjście |

### `autolabel.py` — Grounded SAM → YOLO-seg

Ontologia `{"cube": "kostka"}`. Kolor klasy z ROI + `ulozenie` (R→`czerwona`=0, G→`zielona`=1). Split **po bagach**.

```bash
# Colab GPU zalecany
pip install autodistill autodistill-grounded-sam supervision
python3 tools/autolabel.py --index data/index.csv --visualize
```

Wyjście: `datasets/cubes_seg/{images,labels}/{train,val}/`, `data.yaml`, opcjonalnie `visualizations/`.

### `train_yolo.py` — trening Ultralytics

`ultralytics` jest w obrazie (z ograniczeniem `numpy<2`, patrz wyżej).

```bash
python3 tools/train_yolo.py
python3 tools/train_yolo.py --epochs 100 --device 0 --batch 8 \
  --data datasets/cubes_seg/data.yaml --model yolo11n-seg.pt
```

Wagi: `runs/seg/cubes/weights/best.pt`.

---

## Moduły Pythona

### `core.types`

| Typ | Pola |
|-----|------|
| `Frame` | `rgb` HxWx3 uint8, `depth_m` HxW float32, `K` 3×3, `stamp` |
| `Detection` | `slot`, `label` (surowa nazwa z segmentera), `score`, `mask` HxW bool |
| `Pose` | `position` (3,), `quat` xyzw (4,), `score` |
| `SlotResult` | `slot`, `label` (kanoniczna), `score`, `detection \| None`, `pose \| None` |
| `Segmenter` | `segment(frame) -> list[Detection]` |
| `PoseEstimator` | `estimate(frame, det) -> Pose \| None` |

Etykiety kanoniczne (stałe): `LABEL_RED='red'`, `LABEL_GREEN='green'`, `LABEL_EMPTY='empty'`, `LABEL_UNCERTAIN='uncertain'`.

### `core.dataset`

```python
from g1_perception.core.dataset import load_frame
frame = load_frame(row)  # wiersz z data/index.csv
```

### `core.config`

```python
from g1_perception.core.config import load_rois, load_hsv, load_label_map
rois = load_rois()
hsv = load_hsv()
label_map = load_label_map()  # {'red cube': 'red', 'zielony': 'green', ...}
```

### `core.geometry`

| Funkcja | Opis |
|---------|------|
| `deproject(mask, depth_m, K)` | Nx3 punkty (pomija zera; opcjonalny zakres Z) |
| `fit_plane_ransac(points)` | `(normal, d, n_inliers)` |
| `table_plane_around(roi, mask, …)` | płaszczyzna stołu wokół ROI bez maski obiektu |
| `cube_pose_on_table(points, plane, size=0.05)` | `Pose` (minAreaRect, yaw % 90°, środek + size/2 wzdłuż normalnej) |

### `pipeline`

```python
import g1_perception.segmenters  # rejestruje hsv, yolo_seg
import g1_perception.pose        # rejestruje centroid, geometric
from g1_perception.pipeline import Pipeline

pipe = Pipeline.from_config({
    # opcjonalne; domyślnie config/rois.yaml i config/labels.yaml
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
# albo YOLO:
# 'segmenter': {'name': 'yolo_seg', 'params': {
#     'model_path': 'runs/seg/cubes/weights/best.pt',
#     'verify_hsv': True,
# }}
results = pipe.process(frame)  # list[SlotResult], po jednym na ROI, posortowane po slot
```

Reguły `process`:

- dla każdego ROI z `rois.yaml` dokładnie jeden `SlotResult`;
- brak detekcji w ROI → `empty` (`score=0`, `detection=None`);
- kilka detekcji w jednym slocie → wygrywa najwyższy `score`; detekcje bez slotu są pomijane;
- etykieta mapowana przez `labels.yaml`; nazwa spoza słownika → `uncertain`;
- poza liczona tylko dla `red` / `green` (dla `empty` / `uncertain` `pose=None`).

`from_config` ładuje ROI i mapę etykiet raz i przekazuje je etapom, których konstruktor przyjmuje `rois` / `label_map` (chyba że w `params` podano je jawnie albo `rois_path` / `labels_path`).

### `core.voting`

```python
from g1_perception.core.voting import SlotVoter

voter = SlotVoter(window=10, min_ratio=0.7)  # min_frames domyślnie = window
for frame in frames:
    voted = voter.update(pipe.process(frame))  # list[VotedSlot], po slot
```

Per slot w oknie ostatnich `window` klatek:

- etykieta = większość; `uncertain`, gdy udział < `min_ratio`, remis, za mało klatek (< `min_frames`) albo wygrywa `uncertain`; klatki `uncertain` liczą się do mianownika;
- dla `red` / `green`: pozycja = mediana per oś z klatek zgodnych z wynikiem; yaw = mediana z symetrią 90° (średnia kołowa `4·yaw` jako punkt odniesienia, odchyłki zawijane do ±45°, mediana); oś Z = średnia normalna stołu;
- `VotedSlot`: `slot`, `label`, `score` (udział zwycięzcy), `n_frames`, `pose`, `yaw_deg` ∈ [0, 90), `yaw_consistency` ∈ [0, 1] (1 = zgodne kąty).

### Segmentery

| Nazwa | Moduł | Zachowanie |
|-------|-------|------------|
| `hsv` | `segmenters/hsv.py` | Per ROI: maski HSV red/green, erozja, odsetek ≥ `min_frac` → `red`/`green`; poniżej progu brak detekcji (→ `empty`) |
| `yolo_seg` | `segmenters/yolo_seg.py` | YOLO-seg (`.pt`/`.engine`), `retina_masks=True`, slot gdy pokrycie maski z ROI > 50%; zachowuje nazwę klasy z modelu; opcjonalnie `verify_hsv` (niezgodność z HSV → `uncertain`) |

Import `ultralytics` tylko wewnątrz `YoloSegSegmenter` (opcjonalna zależność).

### Estymatory pozy

| Nazwa | Moduł | Zachowanie |
|-------|-------|------------|
| `centroid` | `pose/centroid.py` | Mediana punktów 3D; orientacja z normalnej stołu |
| `geometric` | `pose/geometric.py` | `cube_pose_on_table` (RANSAC stołu + minAreaRect) |

---

## Dane i topici

**`/bags/labels.csv`:** `bag`, `ulozenie`, `swiatlo`, `czas_s`, `uwagi`, `yaw_deg`

**Topici:**

- `/camera/camera/color/image_raw` (`rgb8`)
- `/camera/camera/aligned_depth_to_color/image_raw` (`16UC1`, mm)
- `/camera/camera/color/camera_info`
- `/tf_static`

**Ułożenie:** `R`/`G`/`X` od lewej (np. `RGGR`).

---

## Konfiguracja YAML

`src/g1_perception/config/` (lokalna kalibracja — zwykle nie commitować):

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

**`labels.yaml`** (w repo) — jedyne miejsce mapowania nazw klas z modeli na etykiety kanoniczne. Klucze bez rozróżniania wielkości liter; wartości tylko `red` / `green` / `empty` / `uncertain`. Nowy model z innymi nazwami klas = nowy wpis tutaj.

```yaml
label_map:
  red cube: red
  czerwona: red
  green cube: green
  zielony: green
  brak: empty
  nieznany: uncertain
```

---

## Testy jednostkowe (mapa)

| Plik | Zakres |
|------|--------|
| `test_pipeline.py` | rejestr, `from_config`, 1 wynik/ROI (4 ROI + 2 detekcje YOLO → 2 kolory + 2 `empty`), `uncertain` |
| `test_labels.py` | `canonical_label`, `load_label_map`, domyślny `labels.yaml` |
| `test_voting.py` | `SlotVoter` (większość, `min_ratio`, remis, okno), mediana pozycji, yaw przez 0/90 |
| `test_frame_utils.py` | głębia mm → m, `32FC1` NaN → 0, `K`, `transform_pose` |
| `test_debug_vis.py` | suma masek mono8, obraz debug |
| `test_node_config.py` | parametry węzła → config pipeline / voter, przebieg z HSV |
| `test_ros_conversions.py` | kontrakt `Detection3DArray` (pomijany bez `vision_msgs`) |
| `test_dataset.py` | `load_frame` |
| `test_config.py` | ROI / HSV YAML |
| `test_hsv_segmenter.py` | segmenter HSV (syntetyki) |
| `test_pose_geometry.py` | geometry + geometric (<5 mm, yaw <2°, szum 2 mm) |
| `test_yolo_seg.py` | sloty / `verify_hsv` (bez ultralytics) |
| `test_autolabel_helpers.py` | split bagów, YOLO polygon (bez GPU) |

---

## Gitignore

- `data/`, `datasets/`, `results/`, `runs/`
- `build/`, `install/`, `log/`
- `*.mcap`, wagi (`*.pt`, `*.onnx`, `*.engine`, …)
