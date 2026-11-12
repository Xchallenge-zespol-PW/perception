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
| `core.voting` | `SlotVoter`: głosowanie per slot w oknie czasowym (sekundy) albo N klatek |
| `core.tracker` | śledzenie 3D (najbliższy sąsiad) + głosowanie per track |
| `core.slot_assign` | numery miejsc 1..N wzdłuż osi + walidacja linii/odstępów |
| `core.cube_registry` | ćwiczenie 1: zamrożenie układu startowego, stałe nazwy `cube_N`, odnajdywanie zgubionej kostki |
| `core.dataset` | `load_frame` z `data/index.csv` |
| `core.geometry` | deprojekcja, RANSAC płaszczyzny, poza kostki, `transform_pose` |
| `core.keypoints` | kanoniczna kolejność 4 narożników górnej ściany (`c0..c3`) |
| `core.pnp` | poza górnej ściany z 4 narożników (IPPE_SQUARE), wybór rozwiązania po normalnej stołu, yaw na stole, środek bryły |
| `core.frame_utils` | głębia mm → m, `K` z `camera_info` |
| `core.debug_vis` | obraz debug (ROI / id tracka, maski, etykiety, yaw), suma masek mono8 |
| `pipeline` | `@register`, `Pipeline.from_config` / `process` (`rois_mode`) |
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
│   ├── config/                    # rois, labels, params_{laptop,robot}.yaml
│   ├── launch/perception.launch.py  # config:=params_laptop.yaml (domyślnie)
│   ├── launch/apriltag.launch.py    # config:=apriltag.yaml
│   ├── launch/balls.launch.py       # config:=balls.yaml (etap 2)
├── deploy/                        # requirements-robot.txt, constraints.txt (numpy<2)
├── docs/robot.md                  # Jetson Orin NX / Humble / TensorRT GPU + ONNX CPU
│   ├── g1_perception/
│   │   ├── core/
│   │   │   ├── types.py
│   │   │   ├── dataset.py
│   │   │   ├── config.py
│   │   │   ├── labels.py
│   │   │   ├── voting.py
│   │   │   ├── frame_utils.py
│   │   │   ├── debug_vis.py
│   │   │   ├── geometry.py
│   │   │   ├── apriltag.py        # AprilTag: aruco + IPPE
│   │   │   ├── ball.py            # środek i promień piłki z maski + głębi
│   │   │   ├── tf_chain.py        # TF z /tf w bagu (offline)
│   │   │   └── visibility.py      # bramki aktualizacji pozy (ucięta / zasłonięta kostka, obrót kamery)
│   │   ├── node_config.py
│   │   ├── ros_conversions.py
│   │   ├── perception_node.py     # entry point: perception_node
│   │   ├── apriltag_node.py       # entry point: apriltag_node
│   │   ├── ball_node.py           # entry point: ball_node (etap 2)
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
│   ├── record_balls.sh            # piłki (etap 2) → labels_pilki.csv
│   ├── labels_csv.py
│   ├── extract_frames.py
│   ├── apriltag_from_bag.py
│   ├── roi_picker.py
│   ├── hsv_tuner.py
│   ├── autolabel.py               # Grounded SAM → YOLO-seg dataset
│   ├── cvat_to_yolo.py            # CVAT (poligon + 4 narożniki) → cubes_seg + cubes_pose
│   ├── preview_pose_labels.py     # kontrola kolejności narożników + podgląd
│   ├── train_yolo.py              # Ultralytics seg / pose (YOLO26)
│   ├── export_model.py            # .pt → .onnx (CPU) / .engine (GPU, na Jetsonie)
│   ├── bench_models.py            # ms / klatkę per model i urządzenie
│   ├── compare_backends.py        # .engine (GPU) vs .onnx (CPU): zgodność wyników
│   └── jetson_gpu_check.py        # gotowość kontenera robota do GPU
├── data/                          # klatki + index.csv (gitignore)
├── datasets/                      # cubes_seg/, cubes_pose/ (gitignore)
├── runs/                          # trening YOLO (gitignore)
└── results/                       # ewaluacja (gitignore)
```

Poza repo (bind mount): `/bags/` — MCAP + `labels.csv`.

## Uruchomienie środowiska

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to g1_perception
source install/setup.bash
```

`g1_perception` zależy od `xchallenge_msgs` (wiadomości zadań, osobny pakiet w workspace X-Challenge). W devcontainerze skopiuj go do `src/` (ścieżka jest w `.gitignore`); `--packages-up-to` zbuduje go przed `g1_perception`. Na robocie kopiuje go `tools/deploy_robot.sh`.

```bash
cp -r <workspace X-Challenge>/src/xchallenge_msgs src/
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

Kamera ~15 kl/s, inferencja na CPU ~125 ms (~8 kl/s): callback synchronizatora zapisuje **tylko najnowszy** zestaw (kolor + głębia + `camera_info`, QoS sensor data). Osobny wątek bierze ten zestaw, buduje `Frame` (rgb8→RGB; YOLO wewnątrz robi BGR; 16UC1 mm→float32 m), woła `Pipeline.process` + `SlotVoter`. Klatki, które przyszły w trakcie inferencji, są **pomijane** (brak kolejki = brak narastającego opóźnienia). Publikacja `/perception/cubes`, maski i debugu idzie z timerem `publish_rate_hz`. Model jest ładowany i rozgrzewany przy starcie węzła (z fallbackiem na CPU, patrz niżej). Co 10 s log: średni czas inferencji [ms], liczba pominiętych klatek i aktywny backend. Przed Ultralytics ustawiane jest `YOLO_AUTOINSTALL=false`.

### Uruchomienie na bagu

Terminal 1 — odtwarzanie w pętli:

```bash
source /opt/ros/humble/setup.bash
ros2 bag play /bags/kostki_RGGR_niskie_20261015_184542 --loop
```

Terminal 2 — węzeł (z `cwd=/ws`, bo `model_path` jest względny):

```bash
cd /ws
source /opt/ros/humble/setup.bash && source install/setup.bash
# laptop (domyślnie config:=params_laptop.yaml):
ros2 launch g1_perception perception.launch.py
# robot:
ros2 launch g1_perception perception.launch.py config:=params_robot.yaml
# nadpisanie ROI bez edycji YAML:
ros2 run g1_perception perception_node --ros-args \
  --params-file install/g1_perception/share/g1_perception/config/params_laptop.yaml \
  -p rois_path:=src/g1_perception/config/rois_RGGR_niskie_184542.yaml
```

Terminal 3 — podgląd:

```bash
ros2 topic echo /perception/cubes
ros2 run rqt_image_view rqt_image_view /perception/debug_image
```

Uwagi:

- `--loop` cofa znaczniki czasu — zostaw `use_sim_time:=false`. `use_sim_time:=true` tylko z `ros2 bag play --clock` bez `--loop`.
- `rois.yaml` / `rois_path` muszą pasować do kadru. ROI w pikselach wymagają stałej pozycji kamery względem stołu.
- `config/*.yaml` instalują się do `share/g1_perception/config/` (`setup.py` glob).
- Na robocie domyślnie TensorRT na GPU; na fallbacku ONNX CPU ~124 ms/klatkę. `skipped_frames` w logu co 10 s pokazuje, ile klatek 15 Hz odrzucono.

### Parametry (`config/params_*.yaml`)

Te same klucze: `params_laptop.yaml` (`.pt` / `.onnx`, `device: cpu`), `params_robot.yaml` (`.engine`, `device: "0"`, fallback `.onnx` na CPU). Wdrożenie G1/PC2: [docs/robot.md](docs/robot.md), `deploy/requirements-robot.txt`.

```bash
ros2 launch g1_perception perception.launch.py config:=params_laptop.yaml
# robot, wymuszenie CPU bez edycji YAML:
ros2 launch g1_perception perception.launch.py config:=params_robot.yaml \
  model_path:=/ws/models/cubes_seg.onnx device:=cpu
```

| Grupa | Parametry |
|-------|-----------|
| Wejście | `color_topic`, `depth_topic` (aligned, `16UC1` mm lub `32FC1` m), `camera_info_topic`, `sync_queue_size`, `sync_slop_s` |
| Wyjście | `output_topic` (`/perception/cubes`), `debug_image_topic`, `mask_topic`, `target_frame` (`""` = ramka kamery), `tf_timeout_s` |
| Pipeline | `segmenter` (`yolo_seg` \| `hsv`), `pose_estimator` (`geometric` \| `centroid` \| `pnp`), `model_path`, `device`, `fallback_model_path` / `fallback_device` (gdy `model_path` nie ruszy na `device`), `warmup_runs` (predykcje na pustym obrazie przy starcie), `conf`, `verify_hsv`, `rois_mode` (`none` \| `image_roi`), `rois_path` / `hsv_path` / `labels_path` (`""` = domyślne; w `none` brak pliku ROI nie jest błędem), `duplicate_overlap` (w `none`: maska leżąca w większej części niż ta na masce wykrycia z wyższym wynikiem, liczonej od mniejszej z nich, to ta sama kostka widziana drugi raz, np. w obu kolorach albo cała i sama górna ściana, i odpada niezależnie od klasy; 0 wyłącza) |
| Poza | `cube_size_m`, `margin_px` (w `none` z `geometric`: jedna płaszczyzna stołu na klatkę, z `margin_px` wokół każdego wykrycia bez masek wszystkich wykryć; wysokość kostki to płaszczyzna + `cube_size_m`/2, więc płaszczyzna liczona osobno dla każdej kostki, bez masek sąsiadów, kładła środkową kostkę kolumny na górnych ścianach sąsiadów, 5 cm za wysoko), `z_min_m`, `z_max_m` (≤ 0 wyłącza); kształt (`geometric`): `min_height_m` (do pozy idą tylko punkty maski od tej wysokości do 1,5 × `cube_size_m` nad stołem; ≤ 0 wyłącza), `min_above_ratio` (mniejsza część punktów nad stołem = płaskie, np. taśma, odrzucone), `footprint_min_m` / `footprint_max_m` (prostokąt na stole; 0 wyłącza) |
| PnP (`pose_estimator: pnp`) | `kpt_min_conf`, `pnp_max_reproj_px`, `pnp_max_depth_diff_m` (przy `pnp_depth_diff_ref_m`, dalej × (z / ref)²), `pnp_max_normal_deg`, `pnp_fallback` (`geometric` \| `none`) |
| Głosowanie / tracking | `vote_window_s` (> 0 = okno czasowe w sekundach stempli klatek; 0 = okno klatek), `vote_min_votes` (min. głosów w oknie czasowym), `vote_window` (okno klatek; w trybie czasowym limit głosów), `vote_min_ratio`, `vote_min_frames` (0 = `vote_window`), `match_dist_m` (w 3D, a z `match_max_dz_m` > 0 w x, y przy różnicy wysokości do `match_max_dz_m`: wysokość źle policzona w jednej klatce nie zakłada nowego tracka, a mediana ją pomija; wykrycie tak blisko tracka zajętego albo założonego w tej samej klatce to druga maska tej kostki i nie zakłada tracka), `max_missed_s` (> 0 = track znika po tylu sekundach bez detekcji) / `max_missed` (klatki), `fresh_s` (> 0 = do miejsc i zamrożenia liczą się tylko tracki widziane w ostatnich tylu sekundach; rozjechany track nie robi z jednej kostki dwóch) |
| Miejsca (`rois_mode=none`) | `slot_axis` (kolejność numerów: miejsce 1 = najmniejsza współrzędna; na robocie `x`, `cube_1` najbliżej), `expected_count`, `max_line_deviation_m`, `min_spacing_m` / `max_spacing_m` (odstęp między sąsiednimi kostkami mierzony wzdłuż dopasowanej linii rzędu, więc rząd może leżeć pod kątem do `slot_axis`); przy większej liczbie tracków (do 8) przechodzi, gdy dokładnie jedna `expected_count`-ka spełnia te warunki, a nadmiarowe tracki są pomijane |
| Ćwiczenie 1 (`rois_mode=none`) | `cube_colors_topic`, `cube_name_prefix`, `start_frame` (`""` = `target_frame`), `latch_s` (ile sekund stempli układ musi być stabilny przed zamrożeniem), `reid_gate_m` |
| Tempo | `publish_rate_hz` |

`params_robot.yaml` używa `rois_mode: none` (YOLO → tracker 3D → TF → numery miejsc) i ustawień ćwiczenia 1: `target_frame: left_ankle_roll_link` (stopa stoi w miejscu; miednica przesuwa się o ok. 2 cm przy obrocie pasa i kołysze przy pchnięciu), `slot_axis: x` (kolumna kostek wzdłuż widoku robota, środki co 6 cm, `cube_1` najbliżej), `start_frame: left_ankle_roll_link`. `params_laptop.yaml` zostaje w `camera_link` z `slot_axis: y`.

### Kontrakt wiadomości

**`/perception/cubes`** — `vision_msgs/Detection3DArray`:

- **`rois_mode=image_roi`**: jedna `Detection3D` na każde ROI, w kolejności slotów (`id` = `"0"`…).
- **`rois_mode=none`**: jedna `Detection3D` na śledzoną kostkę. Po udanej walidacji `id` = numer miejsca `"1"`…`"N"` (sort wzdłuż `slot_axis` w `target_frame`). Gdy walidacja nie przejdzie (np. kostki rozrzucone, nie w rzędzie): `id` = `"t<id trackera>"`, kolor i poza z głosowania zostają, w logu ostrzeżenie.

| Pole | Wartość |
|------|---------|
| `header.stamp` | stempel obrazu kolorowego przetworzonej klatki |
| `header.frame_id` | `target_frame` (tracker śledzi kostki już w tej ramce, bo kamera rusza się z tułowiem); ramka kamery, gdy `target_frame` puste. Przy `rois_mode=none` i braku TF klatka nie jest śledzona: pusta lista w ramce kamery, warning w logu |
| `detections[i].header` | jak `header` |
| `detections[i].id` | numer miejsca / slotu albo `t<id śledzenia>` (gdy assignment nie przeszedł) |
| `detections[i].results` | dokładnie 1 element |
| `results[0].hypothesis.class_id` | `red` \| `green` \| `empty` \| `uncertain` |
| `results[0].hypothesis.score` | udział klatek w oknie zgodnych z wynikiem, [0, 1] |
| `results[0].pose.pose` | poza kostki (środek, kwaternion xyzw; +Z = normalna stołu, yaw modulo 90°) |
| `results[0].pose.covariance` | zera (nieużywane) |
| `bbox.center` | jak `results[0].pose.pose` |
| `bbox.size` | `cube_size_m` na każdej osi, gdy jest poza; `0`, gdy jej brak |

Poza jest tylko dla `red` / `green`. Dla `empty`, `uncertain` albo koloru bez udanej estymacji pozy: `bbox.size = 0`, a pola pozy są zerowe (orientacja jednostkowa) i należy je ignorować.

**`/perception/debug_image`** — `sensor_msgs/Image` `rgb8`, ramka kamery: ROI (gdy są), maski, `id:etykieta status` (+ yaw), czas inferencji w ms. Publikowany tylko, gdy ktoś subskrybuje.

**`/perception/mask`** — `sensor_msgs/Image` `mono8`, ramka kamery: suma masek detekcji w bieżącej klatce (0 / 255).

### Ćwiczenie 1 (`cube_sorting_node` z `unitree_controllers`)

Tylko w `rois_mode=none`. Logika czyta wszystkie ramki względem `pelvis`, a stół wylicza sama z ramek `<cube>_start` i stóp.

**Zamrożenie układu startowego** (`core.cube_registry`): gdy przez `latch_s` sekund (stempli klatek, więc niezależnie od GPU / CPU) przypisanie miejsc jest poprawne, wszystkie kostki mają głosowany kolor `red`/`green` i nie zmieniają się tracki, miejsca ani kolory, miejsce `n` staje się kostką `<cube_name_prefix>n` (`cube_1`…`cube_4`). Układ jest publikowany raz:

| Wyjście | Zawartość |
|---------|-----------|
| `/tf_static` | `<start_frame> → <cube>_start` dla wszystkich kostek w jednej wiadomości (`StaticTransformBroadcaster` w rclpy trzyma tylko ostatnią); poza z głosowania. Wysyłane przed kolorami |
| `cube_colors_topic` (`/cube_colors`) | `xchallenge_msgs/CubeColors`, reliable + transient local, depth 1: `cube_names[i]` ma kolor `colors[i]`, w kolejności miejsc. Jedna wiadomość na układ: logika zużywa ją na jedną próbę |

Ramki startowe wiszą pod `start_frame`, bo logika czyta start przy każdym sprawdzeniu linii: na robocie to stopa (stoi w miejscu, gdy miednica się kołysze; bez zależności od odometrii). Bez TF `start_frame ← target_frame` układ nie jest zamrażany (warning w logu).

**Bieżące pozy**: `/tf` `<target_frame> → <cube>` co `publish_rate_hz` dla każdej kostki układu, ze stemplem bieżącego czasu; kostka poza kadrem zostaje w ostatniej widzianej pozie. Dodatkowo `<target_frame> → <cube>_seen` tylko dla kostek widzianych w danej klatce, ze stemplem tej klatki: jego wiek mówi, jak świeża jest poza (`cube_sorting_node` czyta właśnie te ramki, `cube_seen_suffix` w `task_1.yaml`). Nazwa trzyma się swojego tracka także wtedy, gdy kostka wyjedzie z linii. Gdy track zniknie (dłoń zasłoni kostkę, kamera obróci się z pasem) albo zagłosuje drugi kolor, nazwę przejmuje nowy track, który leży w `reid_gate_m` od ostatniej pozy wzdłuż linii kostek: pchnięcie idzie w poprzek kolumny, a sąsiednia kostka jest 6 cm dalej. Linia to główna oś układu startowego (`slot_axis` tylko dla układu o rozrzucie poniżej 2 cm), bo przy robocie skręconym o 5° do stołu kostka pchnięta o 30 cm przesuwa się o 2,6 cm wzdłuż `slot_axis`. Log: `cube_N found again as track …`. Gdy stempel się cofnie (bag `--loop`), tracker numeruje tracki od nowa, więc każda nazwa zapomina swój track i jest odnajdywana po pozycji.

**Bramki aktualizacji pozy** (tylko `rois_mode=none`): gdy kostka jest ucięta lub zasłonięta, albo gdy kamera szybko się obraca, klatka nie zmienia pozy kostki. Track dalej żyje (nie znika, nie dostaje nowego numeru) i głosuje kolor, ale pozycja z tej klatki nie trafia do mediany, więc `<cube>` trzyma ostatnią dobrą pozę. Taka klatka nie zakłada też nowego tracka. Liczniki w logu `perception stats`: `hidden_poses` (kostki), `turning_frames` (klatki).

| Parametr | Na robocie | Co |
|----------|-----------|----|
| `pose_border_px` | 3 | maska bliżej krawędzi obrazu niż tyle px = kostka ucięta |
| `min_visible_ratio` | 0.45 | pole maski / ściana kostki w jej odległości. Całe kostki z bagów 11.2026 (model v4, 1529 detekcji): 0.47–1.7, p1 0.61, mediana 0.91 (zależy od kąta patrzenia); 0.45 łapie kostki zasłonięte mniej więcej w połowie i bardziej |
| `max_camera_rot_deg_s` | 20 | obrót kamery w `target_frame` między kolejnymi klatkami (z TF). Do dostrojenia na bagach z `/tf` (`record_cubes.sh` nagrywa go od 11.2026) |

`0` wyłącza daną bramkę.

**`~/reset`** (`std_srvs/Trigger`, `/perception_node/reset`): zapomina układ i tracki. Wywołaj przed każdą kolejną próbą (kolory są losowane przed każdą); nowy układ dostaje nowe ramki startowe i nową wiadomość z kolorami.

Wymagania: kamera podpięta do drzewa TF robota (`pelvis` → ramka optyczna kamery); bez tego klatki nie są śledzone (`/perception/cubes` pusty, debug image pokazuje surowe detekcje) i nic nie jest publikowane dla ćwiczenia 1.

---

## Węzeł AprilTag (`apriltag_node`)

Osobny węzeł, niezależny od kostek: wykrywa AprilTagi w obrazie kolorowym (OpenCV aruco, rodziny `16h5` / `25h9` / `36h10` / `36h11`; działa na OpenCV 4.5 contrib i ≥ 4.7) i liczy pozę każdego tagu z jego 4 rogów (`solvePnP` IPPE, bez głębi). Wykrywanie ~1–2 ms na klatkę 640×480 (laptop).

```bash
ros2 launch g1_perception apriltag.launch.py                       # config/apriltag.yaml
ros2 launch g1_perception apriltag.launch.py tag_size_m:=0.162 target_frame:=pelvis
```

| Wyjście | Co |
|---------|----|
| TF `tag_<id>` | pod ramką optyczną kamery, stempel obrazu. Środek tagu, x w prawo, y w górę, z z nadrukowanej strony (dla tagu na podłodze: w górę) |
| `/apriltag/detections` (`vision_msgs/Detection3DArray`) | `id` = numer tagu, `class_id` = `tag36h11`, `score` = 1 / (1 + błąd reprojekcji px), poza w `target_frame` (gdy jego TF jest znany, inaczej w ramce kamery), `bbox.size` = bok tagu |
| `/apriltag/summary` (`std_msgs/String`) | co klatkę JSON z numerem i odległością kamera–środek tagu: `[{"id": 7, "distance_m": 1.832}]` (`[]` = brak tagów z pozą); odległość zawsze od kamery, niezależnie od `target_frame` |
| `/apriltag/debug_image` | obrysy, numery i osie (x czerwona, y zielona, z niebieska); rysowany tylko przy subskrybencie |
| log co `log_period_s` | `tags: 7: 1.83 m, 3: 2.41 m` (tylko gdy coś widać; `0` = wyłączony) |
| log co `stats_period_s` | `frames`, `with_tags`, `ids`, `detect_ms_avg` |

Parametry (`config/apriltag.yaml`): `family`, `tag_size_m` (bok **czarnego** kwadratu, nie białej ramki), `tag_ids` (`"0,1,2"`, `""` = wszystkie), `max_reproj_px` (gorsze dopasowanie rogów = brak pozy), `target_frame`, `tf_prefix`, `publish_tf`, `image_topic`, `camera_info_topic`, `summary_topic`, `log_period_s`.

Podgląd numerów i odległości: `ros2 topic echo /apriltag/summary`. Odległość zależy od `tag_size_m` (zły bok = odległość przeskalowana o ten sam czynnik).

Nagranie do późniejszej analizy (`tools/apriltag_from_bag.py`) musi mieć TF, inaczej nie da się przeliczyć tagów do ramek robota:

```bash
ros2 bag record -s mcap -o tagi_$(date +%Y%m%d_%H%M%S) \
  /camera/camera/color/image_raw /camera/camera/color/camera_info /tf /tf_static
```

Tagi: druk matowy, płasko, z białą ramką wokół; zmierz bok czarnego kwadratu; każdy tag inne id. Przy chodzeniu rozmycie ruchu psuje rogi: lepsze światło / krótsza ekspozycja kamery.

## Węzeł piłek (`ball_node`, etap 2)

Krok 1: każda piłka widoczna w bieżącej klatce jako TF, bez śledzenia i pamięci. Model ten sam co dla kostek (v4, klasa `red ball`).

```bash
ros2 launch g1_perception balls.launch.py                  # robot: /camera/d435, odom, TensorRT
# laptop z RealSense, bez TF robota:
ros2 launch g1_perception balls.launch.py camera_ns:=/camera/camera world_frame:=none \
  model_path:=/ws/models/cubes_seg_v4_aug.onnx device:=cpu
```

| Wyjście | Opis |
|---------|------|
| TF `ball_seen_<K>` | środek piłki w `world_frame` (`odom`), stempel klatki; `K` = kolejność od najbliższej w tej klatce, **nie** stały numer. Piłka poza kadrem zostawia stary TF (wiek = stempel) |
| `/perception/balls` (`geometry_msgs/PoseArray`) | środki piłek z klatki w tej samej kolejności |
| `/perception/balls/debug_image` | maska, okrąg, `K odległość r=promień`; żółta = ucięta krawędzią kadru (nie publikowana); rysowany tylko przy subskrybencie |
| log co `log_period_s` | `balls: 1: 1.73 m r=3.4 cm, 2: ...` (odległość od kamery) |

Pozycja: środek maski wyznacza promień widzenia, jej pole promień w pikselach `r_px`; mediana głębi z wewnętrznego koła maski to przód piłki `z_s`, środek leży o `r = r_px·z_s/(f − r_px)` dalej. Rozmiar piłki nie jest parametrem: zmierzony `r` sprawdza, czy pozycja ma sens (rzeczywisty promień piłki ≈ wynik z logu).

---

## Typowy workflow offline

```text
1. Nagraj bag                 ./tools/record_cubes.sh ...   (piłki: record_balls.sh)
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

### `record_balls.sh` — nagrywanie baga z piłkami (etap 2)

Wymaga działającej kamery RealSense. Te same topiki co `record_cubes.sh` (kamera + `/tf` + `/tf_static`); prefiks kamery `CAM_NS` (domyślnie `/camera/camera`, na robocie `CAM_NS=/camera/d435`).

```bash
./tools/record_balls.sh <liczba_pilek> <polka> <swiatlo> [czas_s] [uwagi]
```

| Argument | Opis |
|----------|------|
| `liczba_pilek` | `0`–`4` |
| `polka` | `h60`, `h100`, `h60_100` (obie półki) albo `stol` |
| `swiatlo` | np. `dzienne`, `lampa` (litery, cyfry, `_`) |
| `czas_s` | sekundy (domyślnie `10`) |
| `uwagi` | tekst |

```bash
./tools/record_balls.sh 4 h60 lampa
./tools/record_balls.sh 2 h60_100 dzienne 15 "jedna przy krawedzi"
```

Zapis: `/bags/pilki_<liczba>_<polka>_<swiatlo>_<data>/` + wiersz w `/bags/labels_pilki.csv` (`bag, liczba_pilek, polka, swiatlo, czas_s, uwagi`). Błędny argument = koniec bez nagrania i bez etykiety. Klatki: `python3 tools/extract_frames.py --labels /bags/labels_pilki.csv --out-dir data_pilki` (`index.csv` dostaje dodatkowo `liczba_pilek`, `polka`).

### `labels_csv.py` — helper CSV

```bash
python3 tools/labels_csv.py ensure /bags/labels.csv
python3 tools/labels_csv.py append /bags/labels.csv \
  --bag NAME --ulozenie RGGR --swiatlo dzienne --czas-s 15 \
  --uwagi "" --yaw-deg "0,15,45,30"
python3 tools/labels_csv.py ensure-balls /bags/labels_pilki.csv
python3 tools/labels_csv.py append-balls /bags/labels_pilki.csv \
  --bag NAME --liczba-pilek 3 --polka h60 --swiatlo lampa --czas-s 10 --uwagi=""
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
data/frames/<bag>/<bag>_<nr>_color.png
data/frames/<bag>/<bag>_<nr>_depth.npy  # float32 [m], 0 = brak
data/frames/<bag>/camera_info.json
data/index.csv                        # + ulozenie, swiatlo, yaw_deg
```

### `apriltag_from_bag.py` — pozy AprilTagów z bagów

Biblioteka **rosbags** (bez `rclpy`; bagi MCAP i sqlite3). Dla każdej klatki (co `--every`) i każdego tagu z pozą zapisuje wiersz w ramce kamery oraz po jednym w każdej ramce z `--frames`, przeliczonej z `/tf` + `/tf_static` z baga (najbliższa próbka TF na każdej krawędzi, bez interpolacji, maks. `--max-tf-dt-ms`).

```bash
python3 tools/apriltag_from_bag.py /bags/tagi_* --tag-size 0.162 --frames pelvis,left_ankle_roll_link
```

| Flaga | Domyślnie | Opis |
|-------|-----------|------|
| `--tag-size` | `0.10` | bok czarnego kwadratu [m] |
| `--family` / `--ids` | `36h11` / wszystkie | rodzina, filtr numerów (`0,1,2`) |
| `--frames` | — | dodatkowe ramki TF |
| `--every` | `1` | co którą klatkę |
| `--max-reproj-px` | `2.0` | jak w węźle |
| `--max-tf-dt-ms` | `50` | max odległość próbki TF od stempla obrazu |
| `--out` | `data/apriltags.csv` | wyjście |

Kolumny: `bag, stamp_s, tag_id, frame, x, y, z, qx, qy, qz, qw, distance_m, reproj_px, ambiguity` (`ambiguity` blisko 1 = tag mógłby być pochylony w drugą stronę, poza niepewna; mały / daleki tag).

### `roi_picker.py` — miejsca kostek

```bash
python3 tools/roi_picker.py data/frames/<bag>/<bag>_0_color.png
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
python3 tools/train_yolo.py                       # seg: datasets/cubes_seg, yolo26n-seg.pt
python3 tools/train_yolo.py --task pose --device 0  # pose: datasets/cubes_pose, yolo26n-pose.pt
python3 tools/train_yolo.py --task pose --model yolo26s-pose.pt --epochs 150
```

`hsv_h=0` (kolor to klasa), dla pose `fliplr=0`; ścieżki datasetu są rozwiązywane do bezwzględnych. Wagi: `runs/seg/cubes/weights/best.pt`, `runs/pose/cubes_pose/weights/best.pt`.

Pełny przepływ danych i modeli (CVAT → datasety → trening → eksport → robot): [docs/training.md](docs/training.md).

---

## Moduły Pythona

### `core.types`

| Typ | Pola |
|-----|------|
| `Frame` | `rgb` HxWx3 uint8, `depth_m` HxW float32, `K` 3×3, `stamp` |
| `Detection` | `slot`, `label` (surowa nazwa z segmentera), `score`, `mask` HxW bool, opcjonalnie `keypoints` Kx3 `(u, v, conf)` (narożniki `c0..c3` z `yolo_pose`) |
| `Pose` | `position` (3,), `quat` xyzw (4,), `score`, opcjonalnie `method` (`pnp` \| `fallback:<powód>`) |
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
| `quad_mask(corners, shape)` | HxW bool wypełnionego wielokąta narożników (Nx2 albo Kx3); dla `yolo_pose` tylko górna ściana |
| `fit_plane_ransac(points)` | `(normal, d, n_inliers)` |
| `table_plane_around(roi, mask, …)` | płaszczyzna stołu wokół ROI bez maski obiektu |
| `table_plane_around_rois(rois, exclude, …)` | jedna płaszczyzna stołu wokół wielu ROI bez pikseli `exclude` (maski wszystkich wykryć) |
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

# okno czasowe: te same decyzje przy 1 i 20 kl/s
voter = SlotVoter(window=100, min_ratio=0.7, window_s=2.0, min_votes=5)
voted = voter.update(pipe.process(frame), frame.stamp)
```

Per slot w oknie ostatnich `window` klatek albo, z `window_s`, głosów ze stemplem nie starszym niż `window_s` od najnowszej klatki (wymaga `min_votes`; stempel cofnięty, np. `ros2 bag play --loop`, czyści historię):

- etykieta = większość; `uncertain`, gdy udział < `min_ratio`, remis, za mało klatek (< `min_frames`) albo wygrywa `uncertain`; klatki `uncertain` liczą się do mianownika;
- dla `red` / `green`: pozycja = mediana per oś z klatek zgodnych z wynikiem; yaw = mediana z symetrią 90° (średnia kołowa `4·yaw` jako punkt odniesienia, odchyłki zawijane do ±45°, mediana); oś Z = średnia normalna stołu;
- `VotedSlot`: `slot`, `label`, `score` (udział zwycięzcy), `n_frames`, `pose`, `yaw_deg` ∈ [0, 90), `yaw_consistency` ∈ [0, 1] (1 = zgodne kąty).

### Segmentery

| Nazwa | Moduł | Zachowanie |
|-------|-------|------------|
| `hsv` | `segmenters/hsv.py` | Per ROI: maski HSV red/green, erozja, odsetek ≥ `min_frac` → `red`/`green`; poniżej progu brak detekcji (→ `empty`) |
| `yolo_seg` | `segmenters/yolo_seg.py` | YOLO-seg (`.pt`/`.onnx`/`.engine`, ładowanie przez `segmenters/yolo_backend.py` z fallbackiem CPU; `.engine` bez torcha przez `trt_runtime.py` + `yolo_decode.py`), `retina_masks=True`, slot gdy pokrycie maski z ROI > 50%; zachowuje nazwę klasy z modelu; opcjonalnie `verify_hsv` (niezgodność z HSV → `uncertain`) |

Import `ultralytics` tylko wewnątrz `YoloSegSegmenter` (opcjonalna zależność).

### Estymatory pozy

| Nazwa | Moduł | Zachowanie |
|-------|-------|------------|
| `centroid` | `pose/centroid.py` | Mediana punktów 3D; orientacja z normalnej stołu |
| `geometric` | `pose/geometric.py` | `cube_pose_on_table` (RANSAC stołu + minAreaRect); `estimate_many` (pipeline w `rois_mode=none`): wszystkie wykrycia klatki na jednej płaszczyźnie stołu bez ich masek |
| `pnp` | `pose/pnp.py` | orientacja z 4 narożników (IPPE_SQUARE, rzut na stół), pozycja z głębi w czworokącie; odrzucenie (`no_keypoints`, `kpt_conf`, `no_table`, `no_depth`, `pnp_failed`, `reproj`, `tilt`, `depth_diff`) → `geometric` na górnej ścianie albo `None` |

---

## Dane i topici

**`/bags/labels.csv`:** `bag`, `ulozenie`, `swiatlo`, `czas_s`, `uwagi`, `yaw_deg`

**Topici:**

- `/camera/camera/color/image_raw` (`rgb8`)
- `/camera/camera/aligned_depth_to_color/image_raw` (`16UC1`, mm)
- `/camera/camera/color/camera_info`
- `/tf`, `/tf_static` (`record_cubes.sh`; `/tf` od 11.2026: ruch kamery w bagu, np. do strojenia `max_camera_rot_deg_s`)

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
| `test_cube_registry.py` | zamrożenie układu, śledzenie pchniętej kostki, odnajdywanie po zgubieniu, weto koloru, reset |
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
