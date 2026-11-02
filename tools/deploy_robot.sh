#!/usr/bin/env bash
# Wgrywa repo na PC2 (G1) i buduje g1_perception w kontenerze humble.
#
# Użycie (z katalogu repo / laptopa):
#   ./tools/deploy_robot.sh <model.onnx> [user@host]
#
# Domyślny host: unitree@192.168.123.164
# Kod  → ~/humble_ws/src/perception/   (bez *.pt, *.onnx, models/)
# Model → ~/humble_ws/models/<basename>  (= /ws/models/ w kontenerze)
# Build → ~/g1_humble/humble.sh run … source ROS + colcon build
#
# Opcjonalnie (żeby nie podawać hasła 3×): ssh-copy-id unitree@192.168.123.164

set -euo pipefail

usage() {
  cat <<'EOF'
Użycie: deploy_robot.sh <model.onnx> [user@host]

  model.onnx   lokalna ścieżka modelu ONNX (imgsz 640); kopiowany do
               ~/humble_ws/models/ na robocie (= /ws/models/ w kontenerze)
  user@host    domyślnie unitree@192.168.123.164

Przykład:
  ./tools/deploy_robot.sh models/cubes_seg.onnx
  ./tools/deploy_robot.sh runs/segment/.../best.onnx unitree@192.168.123.164

Zależności Pythona skrypt NIE instaluje. Na robocie (hotspot, do resetu):
  ~/g1_humble/humble.sh shell
  pip3 install --user \
    -c /ws/src/perception/deploy/constraints.txt \
    -r /ws/src/perception/deploy/requirements-robot.txt
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_PATH="${1:?Podaj ścieżkę do modelu .onnx (deploy_robot.sh <model.onnx> [host])}"
HOST="${2:-unitree@192.168.123.164}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "BLAD: nie ma pliku modelu: $MODEL_PATH" >&2
  exit 1
fi
if [[ "$MODEL_PATH" != *.onnx ]]; then
  echo "BLAD: oczekiwano pliku .onnx, dostano: $MODEL_PATH" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_NAME="$(basename "$MODEL_PATH")"

REMOTE_CODE="humble_ws/src/perception"
REMOTE_MODELS="humble_ws/models"

echo "==> Host:  $HOST"
echo "==> Kod:   $REPO_ROOT/  ->  ${HOST}:${REMOTE_CODE}/"
echo "==> Model: $MODEL_PATH  ->  ${HOST}:${REMOTE_MODELS}/${MODEL_NAME}"

ssh "$HOST" "mkdir -p '${REMOTE_CODE}' '${REMOTE_MODELS}'"

rsync -avz --delete \
  --exclude '.git/' \
  --exclude 'data/' \
  --exclude 'datasets/' \
  --exclude 'runs/' \
  --exclude 'results/' \
  --exclude 'build/' \
  --exclude 'install/' \
  --exclude 'log/' \
  --exclude 'models/' \
  --exclude '*.pt' \
  --exclude '*.onnx' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  "${REPO_ROOT}/" "${HOST}:${REMOTE_CODE}/"

rsync -avz "$MODEL_PATH" "${HOST}:${REMOTE_MODELS}/${MODEL_NAME}"

echo "==> colcon build w kontenerze humble…"
ssh -t "$HOST" \
  '~/g1_humble/humble.sh run bash -lc "source /opt/ros/humble/setup.bash && cd /ws && colcon build --symlink-install --packages-select g1_perception"'

cat <<EOF

==> Deploy OK.

Na robocie (w kontenerze humble):

  # Kamera (D435i, 640x480 @ 15, aligned depth) — osobny terminal:
  ros2 launch realsense2_camera rs_launch.py \\
    camera_namespace:=camera \\
    enable_color:=true enable_depth:=true align_depth.enable:=true \\
    rgb_camera.color_profile:=640x480x15 \\
    depth_module.depth_profile:=640x480x15

  # Węzeł (params_robot.yaml: model_path=/ws/models/cubes_seg.onnx):
  source /opt/ros/humble/setup.bash && source /ws/install/setup.bash
  export YOLO_AUTOINSTALL=false
  cd /ws
  ros2 launch g1_perception perception.launch.py config:=params_robot.yaml

Jeśli wgrano inny plik niż cubes_seg.onnx:
  ros2 launch g1_perception perception.launch.py config:=params_robot.yaml \\
    model_path:=/ws/models/${MODEL_NAME}
EOF
