# Wdrożenie na robocie (Unitree G1 / PC2)

Ustalenia środowiska i uruchamiania `g1_perception` na komputerze towarzyszącym G1.

## Sprzęt i system

| Element | Wartość |
|---------|---------|
| PC2 | Jetson Orin NX |
| JetPack | **5.1.1** |
| Kontener percepcji | ROS 2 **Humble**, Ubuntu **22.04**, Python **3.10** |
| Inferencja | **GPU** (TensorRT `.engine`), fallback **CPU** (ONNX Runtime) |
| Kamera | RealSense **D435i** (głowa), USB **2.1** |
| Profil kamery | **640×480 @ 15** kl/s |
| DDS | **CycloneDDS**, `ROS_DOMAIN_ID=0` |

Isaac ROS / FoundationPose wymagają JetPack 6+ — na PC2 **nie** są używane. Poza = `geometric` (głębia + płaszczyzna stołu); detekcja = YOLO-seg jako silnik **TensorRT na GPU**, z `.onnx` na CPU jako fallbackiem.

## Topici kamery

Te same nazwy na laptopie i na robocie (`config/params_robot.yaml` / `params_laptop.yaml`):

| Topic | Typ / encoding |
|-------|----------------|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image`, **rgb8** |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image`, **16UC1**, milimetry |
| `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` |

## Model na GPU (TensorRT) i fallback CPU (ONNX)

Węzeł przy starcie ładuje `model_path` na `device` i robi `warmup_runs` predykcji na pustym obrazie (Ultralytics otwiera `.onnx` / `.engine` dopiero przy pierwszej predykcji; inicjalizacja TensorRT nie trafia wtedy do pierwszej klatki z kamery). `.engine` działa przez własny backend TensorRT (`segmenters/trt_runtime.py`: bindingi `tensorrt` + `libcudart`, dekodowanie w numpy), bez torcha z CUDA — na JetPacku 5 z Pythonem 3.10 takiego torcha nie ma. Jeśli to się nie uda — brak pliku, brak GPU (`libcuda`) albo bindingów `tensorrt`, błąd TensorRT — ładuje `fallback_model_path` na `fallback_device` i loguje **ostrzeżenie**. Aktywny backend (np. `engine:0` albo `onnx:cpu fallback`) jest w logu startowym, w statystykach co 10 s i na `/perception/debug_image`.

| Parametr | Robot |
|----------|-------|
| `model_path` | **`/ws/models/cubes_seg.engine`** |
| `device` | **`"0"`** |
| `fallback_model_path` | **`/ws/models/cubes_seg.onnx`** |
| `fallback_device` | `cpu` |
| `warmup_runs` | `2` |

- Oba pliki eksportowane przy **`imgsz=640`** (zgodne z profilem kamery / treningiem).
- **`.engine` buduje się na Jetsonie** z `.pt` — silnik jest związany z wersją TensorRT i GPU, plik z laptopa nie zadziała:

```bash
cd /ws
python3 src/perception/tools/export_model.py /ws/models/cubes_seg.pt --formats engine --out /ws/models
```

- Bez torcha z CUDA `export_model.py` buduje silnik przez API TensorRT z tymczasowego `.onnx` (ten sam format pliku co `yolo export`).
- Zgodność silnika z `.onnx` (prostokąty, narożniki, maski; FP16 vs FP32): `python3 src/perception/tools/compare_backends.py /ws/models/cubes_seg.engine /ws/models/cubes_seg.onnx --frames /ws/data/frames` — ma wypisać `OK`.
- Gotowość kontenera do GPU (NumPy 1.x, `cv_bridge`, sterownik CUDA, TensorRT na GPU; torch z CUDA tylko jako INFO; z `--export` także test silnika YOLO26 bez torcha): `python3 src/perception/tools/jetson_gpu_check.py`. Pełny przepływ modeli: [training.md](training.md).

- Wymuszenie CPU bez edycji YAML: `model_path:=/ws/models/cubes_seg.onnx device:=cpu` (argumenty launchu, nie `--ros-args -p`).
- Zmierzone na CPU: **~124 ms / klatkę** w kontenerze humble (Orin NX) przy 640. Na fallbacku głosowanie po oknie klatek trwa odpowiednio dłużej.

Przykład eksportu na laptopie:

```bash
yolo export model=runs/segment/runs/seg/cubes-2/weights/best.pt format=onnx imgsz=640
mkdir -p models
cp runs/segment/runs/seg/cubes-2/weights/best.pt models/cubes_seg.pt
cp runs/segment/runs/seg/cubes-2/weights/best.onnx models/cubes_seg.onnx
./tools/deploy_robot.sh models/cubes_seg.pt models/cubes_seg.onnx
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

- `deploy/constraints.txt` — `numpy<2`, `setuptools>=64,<80`, `packaging>=22`
- `deploy/requirements-robot.txt` — `ultralytics`, `onnxruntime`, `onnx`, `numpy<2`, `opencv-python<4.12`
- Numpy **musi** być 1.x (`cv_bridge`).
- Zły duet `setuptools`/`packaging` w `~/.local` psuje `colcon` (`strip_trailing_zero`). Build w `robot_colcon_build.sh` ustawia `PYTHONNOUSERSITE=1`; na stałe: `pip3 install --user -c … "setuptools>=64,<80" "packaging>=22"`.
- Na stałe: wpisy z `requirements-robot.txt` (z `numpy<2`) do `~/g1_humble/paczki-pip.txt` i `rebuild` uzgodniony z grupą.

Dodaj `YOLO_AUTOINSTALL=false` do `~/.bashrc` użytkownika w kontenerze / entrypointu.

## Wdrożenie kodu (rsync + colcon)

Opcjonalnie raz: `ssh-copy-id unitree@192.168.123.164` (żeby nie podawać hasła 3×).

Z laptopa (kod → `~/humble_ws/src/perception/`, model → `~/humble_ws/models/` = `/ws/models/`):

```bash
./tools/deploy_robot.sh models/cubes_seg.pt models/cubes_seg.onnx
# inny host:
./tools/deploy_robot.sh models/cubes_seg.pt models/cubes_seg.onnx unitree@192.168.123.164
```

Skrypt wyklucza `data/`, `datasets/`, `runs/`, `results/`, `build/`, `install/`, `log/`, `models/`, `.git/`, `*.pt`, `*.onnx`, `*.engine`, potem (bez `bash -lc` — `humble.sh` gubi cudzysłowy; build to `tools/robot_colcon_build.sh`):

```bash
~/g1_humble/humble.sh run bash /ws/src/perception/tools/robot_colcon_build.sh
```

Na końcu wypisuje komendy kamery i węzła (`config:=params_robot.yaml`).

## Laptop vs robot (pliki params)

| | `config/params_laptop.yaml` | `config/params_robot.yaml` |
|--|----------------------------|----------------------------|
| Topici | jak wyżej | jak wyżej |
| `segmenter` / `pose_estimator` | `yolo_seg` / `geometric` | to samo |
| `model_path` | względna `.pt` / `.onnx` | **`/ws/models/cubes_seg.engine`** |
| `device` | `cpu` (laptop bez CUDA) | **`"0"`** (GPU) |
| `fallback_model_path` | `""` (brak) | **`/ws/models/cubes_seg.onnx`** |

Te same klucze parametrów — różnią się wartościami pod środowisko.

## Checklist przed startem na robocie

1. `ROS_DOMAIN_ID=0`, CycloneDDS, kamera publikuje trzy topici przy 640×480×15.  
2. Paczki ROS z `apt`: `ros-humble-vision-msgs` (wiadomości `/perception/cubes`) i `ros-humble-rosbag2-storage-mcap` (nagrania MCAP) — w kontenerze humble wpisane na stałe w `~/g1_humble/paczki-apt.txt`, bo `rebuild` usuwa ręczne instalacje.  
3. `YOLO_AUTOINSTALL=false`, `numpy` 1.x, `cv_bridge` + `onnxruntime` importują się; `python3 src/perception/tools/jetson_gpu_check.py` bez `FAIL` (TensorRT na GPU; torch z CUDA niepotrzebny).  
4. `/ws/models/cubes_seg.engine` (zbudowany na tym Jetsonie) i `/ws/models/cubes_seg.onnx` (imgsz 640) na miejscu.  
5. `colcon build` bez błędów (`--packages-up-to g1_perception`, buduje też `xchallenge_msgs`) + `source /ws/install/setup.bash`.  
6. Kamera podpięta do drzewa TF robota: `ros2 run tf2_ros tf2_echo pelvis camera_color_optical_frame` zwraca transformację.  
7. Launch z `params_robot.yaml`; w logu `Inference backend engine:0` (nie `… fallback`); `/perception/cubes` i opcjonalnie debug image.  
8. Ćwiczenie 1: w logu `start layout frozen: cube_1=…`, `ros2 topic echo /cube_colors --qos-durability transient_local --qos-reliability reliable`, `tf2_echo pelvis cube_1_start`. Przed każdą kolejną próbą `ros2 service call /perception_node/reset std_srvs/srv/Trigger`.

## Wyniki testów na robocie

*(Uzupełniaj po każdym wdrożeniu — punkty 3–5 z checklisty uruchomieniowej.)*

| Data | Backend | Inferencja [ms] | skipped/10s | hz depth | echo classes | uwagi |
|------|---------|-----------------|-------------|----------|--------------|-------|
| | | | | | | |
