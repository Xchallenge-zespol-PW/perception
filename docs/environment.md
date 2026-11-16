# Environment and build

**ROS 2 Humble**, Python 3.10, `numpy<2` (required by `cv_bridge`), developed in a **devcontainer**.

## Build

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to g1_perception
source install/setup.bash
```

`g1_perception` depends on `xchallenge_msgs` (task messages, separate package in the X-Challenge workspace). In the devcontainer, copy it into `src/` (path is in `.gitignore`); `--packages-up-to` builds it before `g1_perception`. On the robot, `tools/deploy_robot.sh` copies it.

```bash
cp -r <X-Challenge workspace>/src/xchallenge_msgs src/
```

Docker image: [`docker/Dockerfile`](../docker/Dockerfile) (ROS Humble + RealSense + OpenCV + rosbags + ultralytics). Pip pins: [`docker/constraints.txt`](../docker/constraints.txt). Devcontainer: [`.devcontainer/`](../.devcontainer/) (`/ws`, `/bags`, host network, udev).

## NumPy < 2 (`cv_bridge` requirement)

Humble’s apt `cv_bridge` is built against NumPy 1.x. With NumPy 2, import fails with `AttributeError: _ARRAY_API not found` (“A module that was compiled using NumPy 1.x cannot be run in NumPy 2…”), so the ROS node will not start. Typical cause: `pip install ultralytics` (e.g. `--user`) without constraints pulls NumPy 2.x and `opencv-python` ≥ 4.12 (which itself needs NumPy 2); packages in `~/.local` then shadow the image versions.

Image safeguards:

- `docker/constraints.txt` (`numpy<2`, `opencv-python<4.12`, `opencv-python-headless<4.12`) is copied to `/etc/pip/constraints.txt` and set as `PIP_CONSTRAINT`, so every `pip install` in the container (including `--user`) is constrained.
- `ultralytics` is installed in the image with those constraints.
- Image build ends with a verification step (fails if NumPy ≥ 2 or `cv_bridge` does not import):

```bash
source /opt/ros/humble/setup.bash
python3 -c "import numpy, cv2; from cv_bridge import CvBridge; assert numpy.__version__.startswith('1')"
```

Fix a running container without rebuild:

```bash
pip3 install --user -c docker/constraints.txt "numpy<2" "opencv-python<4.12" "opencv-python-headless<4.12"
```

The same file pins `setuptools<80`: from 80 onward there is no `setup.py develop`, so `colcon build --symlink-install` ends with `error: option --uninstall not recognized`. Fix: `pip3 install --user -c docker/constraints.txt "setuptools<80"`.

## Tests

```bash
# recommended in this container (anyio conflict / system pytest 6.2)
cd /ws/src/g1_perception && python3 -m pytest -p no:anyio -q && cd /ws

colcon test --packages-select g1_perception
colcon test-result --verbose
```

Do not `pip install --user pytest>=7` — it conflicts with ROS plugins (`launch_testing`).

Test map: [architecture.md](architecture.md#unit-tests-map).

## Artifacts not in git

Local / large artifacts stay out of the repository (see `.gitignore`):

- `data/`, `datasets/`, `results/`, `runs/` — frames, YOLO datasets, training runs, evaluation
- `build/`, `install/`, `log/` — colcon output (local only; not part of the product story)
- `*.mcap` and most weight globs (`*.pt`, `*.onnx`, `*.engine`, …)
- Outside repo (bind mount): `/bags/` — MCAP recordings + `labels.csv`

Weights used for demos/deploy may still live under `models/`; hosting/cleanup is separate from this docs layout.
