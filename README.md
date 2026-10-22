# g1_perception

Percepcja kostek dla robota **Unitree G1 EDU**: na 4 znanych miejscach stołu wykryć kostkę (czerwona / zielona / brak, sześcian ~5 cm) i wyznaczyć jej pozę (pozycja 3D + orientacja).

Środowisko: **ROS 2 Humble**, Python 3.10, `numpy<2` (kompatybilność z `cv_bridge`), praca w **devcontainerze**. Logika percepcji jest niezależna od ROS; węzeł ROS 2 będzie tylko cienką warstwą na końcu.

## Architektura

Pipeline ma wymienne etapy wybierane z konfiguracji:

```text
Frame (RGB + depth + K)
        │
        ▼
   Segmenter  →  list[Detection]
        │
        ▼
 PoseEstimator  →  Pose | None   (per detection)
```

- Typy i protokoły: `g1_perception.core.types`
- Rejestr implementacji + składanie z dict: `g1_perception.pipeline`
- Kalibracja ROI / HSV: `g1_perception.core.config` + YAML w `src/g1_perception/config/`
- Offline dane: `tools/` → `data/` (gitignored)
- Nagrania: MCAP w `/bags`, etykiety w `/bags/labels.csv`

Planowane (jeszcze nie w repo): `segmenters/`, `pose/`, węzeł ROS 2, `evaluate`.

## Struktura repo

```text
/ws
├── docker/Dockerfile          # obraz ROS Humble + RealSense + OpenCV + rosbags
├── .devcontainer/             # montuje /ws i /bags, sieć host, GPU/udev
├── src/g1_perception/         # pakiet ament_python
│   ├── config/                # rois.yaml, hsv.yaml (tworzone narzędziami)
│   ├── g1_perception/
│   │   ├── core/
│   │   │   ├── types.py       # Frame, Detection, Pose, Segmenter, PoseEstimator
│   │   │   ├── dataset.py     # load_frame(row) z data/index.csv
│   │   │   └── config.py      # load/save ROI i HSV
│   │   └── pipeline.py        # @register, Pipeline.from_config / process
│   └── test/                  # pytest
├── tools/                     # skrypty offline (bez rclpy w logice)
├── data/                      # wyekstrahowane klatki + index.csv (gitignore)
└── results/                   # wyniki ewaluacji (gitignore)
```

Poza repo (bind mount w kontenerze):

| Ścieżka | Zawartość |
|---------|-----------|
| `/bags/` | foldery rosbagów MCAP + `labels.csv` |
| `/bags/<bag>/` | `*.mcap` + `metadata.yaml` |

## Uruchomienie środowiska

1. Otwórz repo w VS Code / Cursor → **Reopen in Container**.
2. W kontenerze (katalog `/ws`):

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select g1_perception
source install/setup.bash
```

Przy `--symlink-install` zmiany w `.py` zwykle nie wymagają rebuildu (poza `setup.py` / `package.xml`).

### Testy

```bash
# z /ws, po source install/setup.bash
cd src/g1_perception && python3 -m pytest -q && cd /ws

# albo przez colcon (w tym lintery ament)
colcon test --packages-select g1_perception
colcon test-result --verbose
```

Uwaga: komenda `pytest` może nie być w `PATH` — używaj `python3 -m pytest`.

---

## Typowy workflow offline

```text
1. Nagraj bag          tools/record_cubes.sh
2. Wyciągnij klatki    tools/extract_frames.py
3. Zaznacz ROI         tools/roi_picker.py
4. Ustaw HSV           tools/hsv_tuner.py
5. (dalej) segmentacja / poza / evaluate
```

---

## Narzędzia (`tools/`)

Wszystkie komendy poniżej zakładają `cwd=/ws` oraz `source install/setup.bash` (tam gdzie importują `g1_perception`).

### `record_cubes.sh` — nagrywanie baga

Wymaga działającej kamery RealSense (topici color + aligned depth).

```bash
./tools/record_cubes.sh <ulozenie> <swiatlo> [czas_s] [uwagi] [yaw_deg]
```

| Argument | Opis |
|----------|------|
| `ulozenie` | litery `R` / `G` / `X` od lewej (np. `RGGR`, `RXGR`) |
| `swiatlo` | etykieta warunków (np. `dzienne`, `mocne`) |
| `czas_s` | czas nagrania w sekundach (domyślnie `15`) |
| `uwagi` | dowolny tekst |
| `yaw_deg` | kąty yaw w stopniach, po przecinku; tyle pól ile liter w ułożeniu; dla `X` puste pole |

Przykłady:

```bash
./tools/record_cubes.sh RGGR dzienne 15 "lekki odblask"
./tools/record_cubes.sh RGGR mocne 15 "" "0,15,45,30"
./tools/record_cubes.sh RXGR mocne 15 "" "0,,45,30"
```

Zapisuje MCAP do `/bags/kostki_<ulozenie>_<swiatlo>_<timestamp>/` i dopisuje wiersz do `/bags/labels.csv`.  
Przy pierwszym uruchomieniu dopisze kolumnę `yaw_deg` do starego CSV (puste wartości).

### `labels_csv.py` — helper CSV (używany przez `record_cubes.sh`)

Zwykle nie trzeba odpalać ręcznie:

```bash
python3 tools/labels_csv.py ensure /bags/labels.csv
python3 tools/labels_csv.py append /bags/labels.csv \
  --bag NAME --ulozenie RGGR --swiatlo dzienne --czas-s 15 \
  --uwagi "" --yaw-deg "0,15,45,30"
```

### `extract_frames.py` — klatki z bagów do `data/`

Czyta `/bags/labels.csv` biblioteką **rosbags** (bez `rclpy`). Dla każdego baga z `metadata.yaml` zapisuje co N-tą klatkę kolorową z dopasowaną głębią (max Δt = 20 ms).

```bash
python3 tools/extract_frames.py
python3 tools/extract_frames.py --every 5
python3 tools/extract_frames.py --bags-dir /bags --labels /bags/labels.csv --out-dir data
```

| Flaga | Domyślnie | Opis |
|-------|-----------|------|
| `--every` | `10` | co którą klatkę kolorową brać |
| `--max-dt-ms` | `20` | max różnica czasu kolor↔głębia |
| `--out-dir` | `data` | katalog wyjściowy |

Wyjście:

```text
data/frames/<bag>/<nr>_color.png
data/frames/<bag>/<nr>_depth.npy      # float32, metry; 0 = brak pomiaru
data/frames/<bag>/camera_info.json  # K, width, height
data/index.csv                      # bag, frame, ścieżki, ulozenie, swiatlo, yaw_deg
```

Bagi bez `metadata.yaml` są pomijane z ostrzeżeniem.

### `roi_picker.py` — zaznaczanie miejsc kostek

GUI OpenCV (`cv2.selectROIs`). Kolejność zaznaczeń = `slot` 0, 1, 2, …

```bash
python3 tools/roi_picker.py data/frames/<bag>/0_color.png
python3 tools/roi_picker.py data/frames/<bag>/0_color.png --output src/g1_perception/config/rois.yaml
```

Sterowanie w oknie: przeciągnij prostokąt → **ENTER/SPACE** (kolejny) → **ESC** (koniec).  
Zapis: `src/g1_perception/config/rois.yaml` (lista `{slot, x0, y0, x1, y1}`).

### `hsv_tuner.py` — strojenie progów HSV

Przegląda klatki z `data/index.csv`, nakłada maskę czerwony/zielony i ramki ROI.

```bash
python3 tools/hsv_tuner.py
python3 tools/hsv_tuner.py --index data/index.csv --rois src/g1_perception/config/rois.yaml
```

| Klawisz | Akcja |
|---------|--------|
| `n` / `p` | następna / poprzednia klatka |
| `s` | zapisz `hsv.yaml` |
| `q` / `ESC` | wyjście |

Suwaki: dwa zakresy hue dla czerwonego (`red_low` ~0–10, `red_high` ~170–180) oraz zielony (`green`), każdy z S/V.  
Zapis: `src/g1_perception/config/hsv.yaml`.

---

## Moduły Pythona (`g1_perception`)

### `core.types` — typy danych

| Typ | Pola |
|-----|------|
| `Frame` | `rgb` HxWx3 uint8, `depth_m` HxW float32 (m), `K` 3×3, `stamp` |
| `Detection` | `slot`, `label`, `score`, `mask` HxW bool |
| `Pose` | `position` (3,), `quat` xyzw (4,), `score` |
| `Segmenter` | protokół: `segment(frame) -> list[Detection]` |
| `PoseEstimator` | protokół: `estimate(frame, det) -> Pose \| None` |

### `core.dataset` — ładowanie klatki z indeksu

```python
from g1_perception.core.dataset import load_frame

frame = load_frame(row)              # row = dict z data/index.csv
frame = load_frame(row, root=Path('/ws'))
```

Czyta PNG + `.npy` + `camera_info.json` obok kolorowego obrazu.

### `core.config` — ROI i HSV z YAML

```python
from g1_perception.core.config import load_rois, load_hsv, save_rois, save_hsv

rois = load_rois()       # src/g1_perception/config/rois.yaml
hsv = load_hsv()         # src/g1_perception/config/hsv.yaml
```

Dataclassy: `Roi`, `HsvRange`, `HsvConfig`.

### `pipeline` — rejestr i uruchomienie

```python
from g1_perception.pipeline import Pipeline, register

@register('my_segmenter')
class MySegmenter:
    def __init__(self, **params): ...
    def segment(self, frame): ...

@register('my_pose')
class MyPose:
    def __init__(self, **params): ...
    def estimate(self, frame, det): ...

pipe = Pipeline.from_config({
    'segmenter': {'name': 'my_segmenter', 'params': {}},
    'pose_estimator': {'name': 'my_pose', 'params': {}},
})
results = pipe.process(frame)   # list[tuple[Detection, Pose | None]]
```

---

## Dane i topici

**`/bags/labels.csv`:** `bag`, `ulozenie`, `swiatlo`, `czas_s`, `uwagi`, `yaw_deg`

**Topici w bagach:**

- `/camera/camera/color/image_raw` (`rgb8`)
- `/camera/camera/aligned_depth_to_color/image_raw` (`16UC1`, mm)
- `/camera/camera/color/camera_info`
- `/tf_static`

**Ułożenie:** ciąg `R`/`G`/`X` opisujący kolory od lewej (np. `RGGR`).

---

## Konfiguracja YAML

Pliki w `src/g1_perception/config/` (tworzone narzędziami, nie commituj lokalnych kalibracji bez potrzeby):

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

---

## Gitignore (ważne katalogi)

- `data/`, `results/` — artefakty lokalne
- `build/`, `install/`, `log/` — colcon
- `*.mcap`, wagi modeli (`*.pt`, `*.onnx`, …)
