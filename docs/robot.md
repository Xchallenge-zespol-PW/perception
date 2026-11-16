# Robot deploy (Unitree G1 / PC2)

Deploy `g1_perception` on the G1 companion PC (Jetson Orin NX, JetPack 5.1.1, ROS 2 Humble). Prefer TensorRT `.engine` on GPU; fall back to ONNX on CPU. Camera: RealSense D435i at 640×480 @ 15 Hz.

## Hardware and system

| Item | Value |
|------|-------|
| PC2 | Jetson Orin NX |
| JetPack | **5.1.1** |
| Perception container | ROS 2 **Humble**, Ubuntu **22.04**, Python **3.10** |
| Inference | **GPU** (TensorRT `.engine`), fallback **CPU** (ONNX Runtime) |
| Camera | RealSense **D435i** (head), USB **2.1** |
| Camera profile | **640×480 @ 15** fps |
| DDS | **CycloneDDS**, `ROS_DOMAIN_ID=0` |

Isaac ROS / FoundationPose need JetPack 6+ — **not** used on PC2. Pose = `geometric` (depth + table plane); detection = YOLO-seg as a **TensorRT GPU** engine, with `.onnx` on CPU as fallback.

## Camera topics

Same names on laptop and robot (`config/params_robot.yaml` / `params_laptop.yaml`):

| Topic | Type / encoding |
|-------|-----------------|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image`, **rgb8** |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image`, **16UC1**, millimetres |
| `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` |

## GPU model (TensorRT) and CPU fallback (ONNX)

At start the node loads `model_path` on `device` and runs `warmup_runs` predictions on an empty image (Ultralytics opens `.onnx` / `.engine` only on the first predict; TensorRT init then does not hit the first camera frame). `.engine` runs through a custom TensorRT backend (`segmenters/trt_runtime.py`: `tensorrt` + `libcudart` bindings, numpy decode), without CUDA torch — JetPack 5 with Python 3.10 has no such torch. If that fails — missing file, no GPU (`libcuda`) or `tensorrt` bindings, TensorRT error — it loads `fallback_model_path` on `fallback_device` and logs a **warning**. The active backend (e.g. `engine:0` or `onnx:cpu fallback`) appears in the start log, in the 10 s stats, and on `/perception/debug_image`.

| Param | Robot |
|-------|-------|
| `model_path` | **`/ws/models/cubes_seg.engine`** |
| `device` | **`"0"`** |
| `fallback_model_path` | **`/ws/models/cubes_seg.onnx`** |
| `fallback_device` | `cpu` |
| `warmup_runs` | `2` |

- Both files exported at **`imgsz=640`** (matches camera profile / training).
- **`.engine` must be built on the Jetson** from `.pt` — the engine is tied to TensorRT version and GPU; a laptop file will not work:

```bash
cd /ws
python3 src/perception/tools/export_model.py /ws/models/cubes_seg.pt --formats engine --out /ws/models
```

- Without CUDA torch, `export_model.py` builds the engine via the TensorRT API from a temporary `.onnx` (same file format as `yolo export`).
- Engine vs `.onnx` agreement (boxes, corners, masks; FP16 vs FP32): `python3 src/perception/tools/compare_backends.py /ws/models/cubes_seg.engine /ws/models/cubes_seg.onnx --frames /ws/data/frames` — should print `OK`.
- Container GPU readiness (NumPy 1.x, `cv_bridge`, CUDA driver, TensorRT on GPU; CUDA torch only as INFO; with `--export` also YOLO26 engine test without torch): `python3 src/perception/tools/jetson_gpu_check.py`. Full model flow: [training.md](training.md).

- Force CPU without editing YAML: `model_path:=/ws/models/cubes_seg.onnx device:=cpu` (launch args, not `--ros-args -p`).
- Measured on robot: **TensorRT `.engine` 30.9 ms / frame** vs **ONNX CPU 1494.3 ms / frame**. On fallback, frame-window voting takes correspondingly longer.

Laptop export example:

```bash
yolo export model=runs/segment/runs/seg/cubes-2/weights/best.pt format=onnx imgsz=640
mkdir -p models
cp runs/segment/runs/seg/cubes-2/weights/best.pt models/cubes_seg.pt
cp runs/segment/runs/seg/cubes-2/weights/best.onnx models/cubes_seg.onnx
./tools/deploy_robot.sh models/cubes_seg.pt models/cubes_seg.onnx
```

## Python dependencies (important: NumPy)

Humble `cv_bridge` is built against **NumPy 1.x**. Ultralytics at start can **auto-install / bump `numpy` to 2.x**, which breaks `cv_bridge` import (`_ARRAY_API not found`).

Safeguards:

```bash
export YOLO_AUTOINSTALL=false   # required in the robot container
```

`deploy_robot.sh` does **not** install Python packages. For testing (until reboot), with hotspot:

```bash
~/g1_humble/humble.sh shell
pip3 install --user \
  -c /ws/src/perception/deploy/constraints.txt \
  -r /ws/src/perception/deploy/requirements-robot.txt
python3 -c "import numpy, onnxruntime; from cv_bridge import CvBridge; print(numpy.__version__)"
```

- `deploy/constraints.txt` — `numpy<2`, `setuptools>=64,<80`, `packaging>=22`
- `deploy/requirements-robot.txt` — `ultralytics`, `onnxruntime`, `onnx`, `numpy<2`, `opencv-python<4.12`
- NumPy **must** be 1.x (`cv_bridge`).
- A bad `setuptools`/`packaging` pair in `~/.local` breaks `colcon` (`strip_trailing_zero`). Build in `robot_colcon_build.sh` sets `PYTHONNOUSERSITE=1`; permanently: `pip3 install --user -c … "setuptools>=64,<80" "packaging>=22"`.
- Permanently: add `requirements-robot.txt` entries (with `numpy<2`) to `~/g1_humble/paczki-pip.txt` and rebuild agreed with the team.

Add `YOLO_AUTOINSTALL=false` to the container user’s `~/.bashrc` / entrypoint.

## Code deploy (rsync + colcon)

Optional once: `ssh-copy-id unitree@192.168.123.164` (avoid typing the password 3×).

From the laptop (code → `~/humble_ws/src/perception/`, model → `~/humble_ws/models/` = `/ws/models/`):

```bash
./tools/deploy_robot.sh models/cubes_seg.pt models/cubes_seg.onnx
# other host:
./tools/deploy_robot.sh models/cubes_seg.pt models/cubes_seg.onnx unitree@192.168.123.164
```

The script excludes `data/`, `datasets/`, `runs/`, `results/`, `build/`, `install/`, `log/`, `models/`, `.git/`, `*.pt`, `*.onnx`, `*.engine`, then (without `bash -lc` — `humble.sh` drops quotes; build is `tools/robot_colcon_build.sh`):

```bash
~/g1_humble/humble.sh run bash /ws/src/perception/tools/robot_colcon_build.sh
```

At the end it prints camera and node commands (`config:=params_robot.yaml`).

## Laptop vs robot (params files)

| | `config/params_laptop.yaml` | `config/params_robot.yaml` |
|--|----------------------------|----------------------------|
| Topics | as above | as above |
| `segmenter` / `pose_estimator` | `yolo_seg` / `geometric` | same |
| `model_path` | relative `.pt` / `.onnx` | **`/ws/models/cubes_seg.engine`** |
| `device` | `cpu` (laptop without CUDA) | **`"0"`** (GPU) |
| `fallback_model_path` | `""` (none) | **`/ws/models/cubes_seg.onnx`** |

Same parameter keys — values differ by environment.

## Pre-start checklist on the robot

1. `ROS_DOMAIN_ID=0`, CycloneDDS, camera publishes the three topics at 640×480×15.
2. ROS apt packages: `ros-humble-vision-msgs` (for `/perception/cubes`) and `ros-humble-rosbag2-storage-mcap` (MCAP recordings) — pinned in the humble container via `~/g1_humble/paczki-apt.txt`, because `rebuild` wipes manual installs.
3. `YOLO_AUTOINSTALL=false`, NumPy 1.x, `cv_bridge` + `onnxruntime` import; `python3 src/perception/tools/jetson_gpu_check.py` with no `FAIL` (TensorRT on GPU; CUDA torch not required).
4. `/ws/models/cubes_seg.engine` (built on this Jetson) and `/ws/models/cubes_seg.onnx` (imgsz 640) in place.
5. `colcon build` clean (`--packages-up-to g1_perception`, also builds `xchallenge_msgs`) + `source /ws/install/setup.bash`.
6. Camera in the robot TF tree: `ros2 run tf2_ros tf2_echo pelvis camera_color_optical_frame` returns a transform.
7. Launch with `params_robot.yaml`; log shows `Inference backend engine:0` (not `… fallback`); `/perception/cubes` and optionally debug image.
8. Exercise 1: log `start layout frozen: cube_1=…`, `ros2 topic echo /cube_colors --qos-durability transient_local --qos-reliability reliable`, `tf2_echo pelvis cube_1_start`. Before each next attempt `ros2 service call /perception_node/reset std_srvs/srv/Trigger`.
