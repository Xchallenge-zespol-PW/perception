#!/usr/bin/env bash
# Wgrywa repo na PC2 (G1) i buduje g1_perception w kontenerze humble.
#
# Użycie (z katalogu repo / laptopa):
#   ./tools/deploy_robot.sh <model.onnx> [user@host]
#
# Domyślny host: unitree@192.168.123.164
# Kod  → ~/humble_ws/src/perception/
# Model → ~/humble_ws/models/<basename>
# Build → ~/g1_humble/humble.sh run … colcon build --packages-select g1_perception

set -euo pipefail

usage() {
  cat <<'EOF'
Użycie: deploy_robot.sh <model.onnx> [user@host]

  model.onnx   lokalna ścieżka modelu ONNX (imgsz 640); kopiowany do
               ~/humble_ws/models/ na robocie
  user@host    domyślnie unitree@192.168.123.164

Przykład:
  ./tools/deploy_robot.sh models/cubes_seg.onnx
  ./tools/deploy_robot.sh runs/segment/.../best.onnx unitree@192.168.123.164
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
  --exclude '*.pt' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  "${REPO_ROOT}/" "${HOST}:${REMOTE_CODE}/"

rsync -avz "$MODEL_PATH" "${HOST}:${REMOTE_MODELS}/${MODEL_NAME}"

echo "==> colcon build w kontenerze humble…"
ssh -t "$HOST" \
  '~/g1_humble/humble.sh run bash -lc "cd /ws && colcon build --symlink-install --packages-select g1_perception"'

cat <<EOF

==> Deploy OK.

Na robocie (w kontenerze humble, cwd=/ws):

  # Kamera (D435i, 640x480 @ 15, aligned depth):
  ros2 launch realsense2_camera rs_launch.py \\
    camera_namespace:=camera \\
    enable_color:=true enable_depth:=true align_depth.enable:=true \\
    rgb_camera.color_profile:=640x480x15 \\
    depth_module.depth_profile:=640x480x15

  # Węzeł percepcji (model: models/${MODEL_NAME} względem cwd):
  source /opt/ros/humble/setup.bash && source install/setup.bash
  export YOLO_AUTOINSTALL=false
  ros2 launch g1_perception perception.launch.py config:=params_robot.yaml

Jeśli params_robot.yaml ma inną model_path niż models/${MODEL_NAME}, nadpisz:
  ros2 launch g1_perception perception.launch.py config:=params_robot.yaml \\
    --ros-args -p model_path:=models/${MODEL_NAME}
EOF
