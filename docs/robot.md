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
- Na robocie `model_path` w `params_robot.yaml` wskazuje plik **`.onnx`** (domyślnie `models/cubes_seg.onnx` względem cwd workspace).
- Zmierzone: **~124 ms / klatkę** na CPU w kontenerze humble (Orin NX) przy 640 — miesci się w 15 kl/s kamery przy `publish_rate_hz: 5` (przetwarzana jest najnowsza zsynchronizowana klatka).

Przykład eksportu na laptopie:

```bash
yolo export model=runs/segment/runs/seg/cubes-2/weights/best.pt format=onnx imgsz=640
# skopiuj .onnx na robota jako models/cubes_seg.onnx
```

## Zależności Pythona (ważne: NumPy)

`cv_bridge` z Humble jest skompilowany pod **NumPy 1.x**. Ultralytics przy starcie potrafi **samodzielnie doinstalować / podbić `numpy` do 2.x**, co psuje import `cv_bridge` (`_ARRAY_API not found`).

Zabezpieczenia:

```bash
export YOLO_AUTOINSTALL=false   # obowiązkowe w kontenerze na robocie
pip3 install --user -c deploy/constraints.txt -r deploy/requirements-robot.txt
```

- `deploy/constraints.txt` — `numpy<2`
- `deploy/requirements-robot.txt` — `ultralytics`, `onnxruntime`, `onnx`, `numpy<2`, `opencv-python<4.12`

Weryfikacja:

```bash
python3 -c "import numpy; from cv_bridge import CvBridge; assert numpy.__version__.startswith('1'); print(numpy.__version__)"
```

Dodaj `YOLO_AUTOINSTALL=false` do `~/.bashrc` użytkownika w kontenerze / entrypointu.

## Wdrożenie kodu (rsync + colcon)

Z laptopa (całe repo → `~/humble_ws/src/perception/`, model → `~/humble_ws/models/`, build w kontenerze):

```bash
./tools/deploy_robot.sh models/cubes_seg.onnx
# inny host:
./tools/deploy_robot.sh models/cubes_seg.onnx unitree@192.168.123.164
```

Skrypt wyklucza `data/`, `datasets/`, `runs/`, `results/`, `build/`, `install/`, `log/`, `.git/` i `*.pt`, potem przez SSH:

```bash
~/g1_humble/humble.sh run bash -lc \
  "cd /ws && colcon build --symlink-install --packages-select g1_perception"
```

Na końcu wypisuje komendy kamery i węzła (`config:=params_robot.yaml`). Upewnij się, że `model_path` / `rois_path` pasują do instalacji na G1.

## Laptop vs robot (pliki params)

| | `config/params_laptop.yaml` | `config/params_robot.yaml` |
|--|----------------------------|----------------------------|
| Topici | jak wyżej | jak wyżej |
| `segmenter` / `pose_estimator` | `yolo_seg` / `geometric` | to samo |
| `model_path` | `.pt` (lub `.onnx`) | **`.onnx`**, imgsz 640 |
| `device` | `""` (auto / CUDA) | `cpu` |

Te same klucze parametrów — różnią się wartościami pod środowisko.

## Checklist przed startem na robocie

1. `ROS_DOMAIN_ID=0`, CycloneDDS, kamera publikuje trzy topici przy 640×480×15.  
2. `YOLO_AUTOINSTALL=false`, `numpy` 1.x, `cv_bridge` importuje się.  
3. `models/cubes_seg.onnx` (imgsz 640) na miejscu.  
4. `colcon build` + `source install/setup.bash`.  
5. Launch z `params_robot.yaml`; `/perception/cubes` i opcjonalnie debug image.
