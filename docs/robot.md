# Wdrożenie na robocie (Unitree G1 / PC2)

Ustalenia środowiska i uruchamiania `g1_perception` na komputerze towarzyszącym G1.

## Sprzęt i system

| Element | Wartość |
|---------|---------|
| PC2 | Jetson Orin NX |
| JetPack | **5.1.1** |
| Kontener percepcji | ROS 2 **Humble**, Ubuntu **22.04**, Python **3.10** |
| CUDA w kontenerze humble | **brak** (inferencja na CPU) |
| Kamera | RealSense **D435i** (głowa), USB **2.1** |
| Profil kamery | **640×480 @ 15** kl/s |
| DDS | **CycloneDDS**, `ROS_DOMAIN_ID=0` |

Isaac ROS / FoundationPose wymagają JetPack 6+ — na PC2 **nie** są używane. Poza = `geometric` (głębia + płaszczyzna stołu); detekcja = YOLO-seg przez **ONNX Runtime (CPU)**.

## Topici kamery

Te same nazwy na laptopie i na robocie (`config/params_robot.yaml` / `params_laptop.yaml`):

| Topic | Typ / encoding |
|-------|----------------|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image`, **rgb8** |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image`, **16UC1**, milimetry |
| `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` |

## Model ONNX

- Eksport z wag `.pt` przy **`imgsz=640`** (zgodne z profilem kamery / treningiem).
- Na robocie `model_path` w `params_robot.yaml` to ścieżka bezwzględna **`/ws/models/cubes_seg.onnx`** (`deploy_robot.sh` kopiuje `.onnx` do `~/humble_ws/models/` = `/ws/models/` w kontenerze). Nadpisanie: `model_path:=/ws/models/inny.onnx` (argument launchu, nie `--ros-args -p`).
- Zmierzone: **~124 ms / klatkę** na CPU w kontenerze humble (Orin NX) przy 640 — miesci się w 15 kl/s kamery przy `publish_rate_hz: 5` (przetwarzana jest najnowsza zsynchronizowana klatka).

Przykład eksportu na laptopie:

```bash
yolo export model=runs/segment/runs/seg/cubes-2/weights/best.pt format=onnx imgsz=640
# potem: ./tools/deploy_robot.sh <ścieżka>/best.onnx  →  /ws/models/best.onnx
# albo zapisz lokalnie jako models/cubes_seg.onnx i wgraj pod tą nazwą
```

## Zależności Pythona (ważne: NumPy)

`cv_bridge` z Humble jest skompilowany pod **NumPy 1.x**. Ultralytics przy starcie potrafi **samodzielnie doinstalować / podbić `numpy` do 2.x**, co psuje import `cv_bridge` (`_ARRAY_API not found`).

Zabezpieczenia:

```bash
export YOLO_AUTOINSTALL=false   # obowiązkowe w kontenerze na robocie
```

`deploy_robot.sh` **nie** instaluje pakietów Pythona. Na czas testów (do resetu), z hotspotem:

```bash
~/g1_humble/humble.sh shell
pip3 install --user \
  -c /ws/src/perception/deploy/constraints.txt \
  -r /ws/src/perception/deploy/requirements-robot.txt
python3 -c "import numpy, onnxruntime; from cv_bridge import CvBridge; print(numpy.__version__)"
```

- `deploy/constraints.txt` — `numpy<2`
- `deploy/requirements-robot.txt` — `ultralytics`, `onnxruntime`, `onnx`, `numpy<2`, `opencv-python<4.12`
- Numpy **musi** być 1.x (`cv_bridge`).
- Na stałe: wpisy z `requirements-robot.txt` (z `numpy<2`) do `~/g1_humble/paczki-pip.txt` i `rebuild` uzgodniony z grupą.

Dodaj `YOLO_AUTOINSTALL=false` do `~/.bashrc` użytkownika w kontenerze / entrypointu.

## Wdrożenie kodu (rsync + colcon)

Opcjonalnie raz: `ssh-copy-id unitree@192.168.123.164` (żeby nie podawać hasła 3×).

Z laptopa (kod → `~/humble_ws/src/perception/`, model → `~/humble_ws/models/` = `/ws/models/`):

```bash
./tools/deploy_robot.sh models/cubes_seg.onnx
# inny host:
./tools/deploy_robot.sh models/cubes_seg.onnx unitree@192.168.123.164
```

Skrypt wyklucza `data/`, `datasets/`, `runs/`, `results/`, `build/`, `install/`, `log/`, `models/`, `.git/`, `*.pt`, `*.onnx`, potem:

```bash
~/g1_humble/humble.sh run bash -lc \
  "source /opt/ros/humble/setup.bash && cd /ws && \
   colcon build --symlink-install --packages-select g1_perception"
```

Na końcu wypisuje komendy kamery i węzła (`config:=params_robot.yaml`).

## Laptop vs robot (pliki params)

| | `config/params_laptop.yaml` | `config/params_robot.yaml` |
|--|----------------------------|----------------------------|
| Topici | jak wyżej | jak wyżej |
| `segmenter` / `pose_estimator` | `yolo_seg` / `geometric` | to samo |
| `model_path` | względna `.pt` / `.onnx` | **`/ws/models/cubes_seg.onnx`** |
| `device` | `""` (auto / CUDA) | `cpu` |

Te same klucze parametrów — różnią się wartościami pod środowisko.

## Checklist przed startem na robocie

1. `ROS_DOMAIN_ID=0`, CycloneDDS, kamera publikuje trzy topici przy 640×480×15.  
2. `YOLO_AUTOINSTALL=false`, `numpy` 1.x, `cv_bridge` + `onnxruntime` importują się.  
3. `/ws/models/cubes_seg.onnx` (imgsz 640) na miejscu.  
4. `colcon build` bez błędów + `source /ws/install/setup.bash`.  
5. Launch z `params_robot.yaml`; `/perception/cubes` i opcjonalnie debug image.

## Wyniki testów na robocie

*(Uzupełniaj po każdym wdrożeniu — punkty 3–5 z checklisty uruchomieniowej.)*

| Data | Inferencja [ms] | skipped/10s | hz depth | echo classes | uwagi |
|------|-----------------|-------------|----------|--------------|-------|
| | | | | | |
