#!/usr/bin/env bash
# Wgrywa repo na PC2 (G1) i buduje g1_perception w kontenerze humble.
#
# Użycie (z katalogu repo / laptopa):
#   ./tools/deploy_robot.sh <model> [<model>...] [user@host]
#
# Domyślny host: unitree@192.168.123.164
# Kod    → ~/humble_ws/src/perception/   (bez *.pt, *.onnx, *.engine, models/)
# Modele → ~/humble_ws/models/<basename>  (= /ws/models/ w kontenerze)
# Build  → ~/g1_humble/humble.sh run … source ROS + colcon build
#
# Modele: .pt (z niego powstaje .engine na Jetsonie) i .onnx (fallback CPU).
#
# Opcjonalnie (żeby nie podawać hasła 3×): ssh-copy-id unitree@192.168.123.164

set -euo pipefail

usage() {
  cat <<'EOF'
Użycie: deploy_robot.sh <model> [<model>...] [user@host]

  model        lokalny plik .pt / .onnx / .engine (imgsz 640); kopiowany do
               ~/humble_ws/models/ na robocie (= /ws/models/ w kontenerze)
  user@host    domyślnie unitree@192.168.123.164

Przykład:
  ./tools/deploy_robot.sh models/cubes_seg.pt models/cubes_seg.onnx
  ./tools/deploy_robot.sh runs/segment/.../best.onnx unitree@192.168.123.164

Silnik .engine buduje się na robocie z .pt (docs/robot.md, "Model na GPU");
.engine z innej maszyny zwykle nie zadziała (inna wersja TensorRT / GPU).

Zależności Pythona skrypt NIE instaluje. Na robocie (hotspot, do resetu):
  ~/g1_humble/humble.sh shell
  pip3 install --user \
    -c /ws/src/perception/deploy/constraints.txt \
    -r /ws/src/perception/deploy/requirements-robot.txt
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

HOST="unitree@192.168.123.164"
MODELS=()
for arg in "$@"; do
  case "$arg" in
    *.pt|*.onnx|*.engine)
      if [[ ! -f "$arg" ]]; then
        echo "BLAD: nie ma pliku modelu: $arg" >&2
        exit 1
      fi
      if [[ "$arg" == *.engine ]]; then
        echo "UWAGA: $arg musi być zbudowany na tym Jetsonie (ta sama wersja" \
          "TensorRT), inaczej węzeł przejdzie na fallback CPU." >&2
      fi
      MODELS+=("$arg")
      ;;
    *@*)
      HOST="$arg"
      ;;
    *)
      echo "BLAD: nieznany argument: $arg" \
        "(oczekiwano .pt / .onnx / .engine albo user@host)" >&2
      exit 1
      ;;
  esac
done
if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "BLAD: podaj co najmniej jeden model (.pt / .onnx / .engine)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE_CODE="humble_ws/src/perception"
REMOTE_MODELS="humble_ws/models"

echo "==> Host:  $HOST"
echo "==> Kod:   $REPO_ROOT/  ->  ${HOST}:${REMOTE_CODE}/"
for model in "${MODELS[@]}"; do
  echo "==> Model: $model  ->  ${HOST}:${REMOTE_MODELS}/$(basename "$model")"
done

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
  --exclude '*.engine' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  "${REPO_ROOT}/" "${HOST}:${REMOTE_CODE}/"

rsync -avz "${MODELS[@]}" "${HOST}:${REMOTE_MODELS}/"

echo "==> colcon build w kontenerze humble…"
# Do NOT use bash -lc 'source … && colcon …': humble.sh strips quotes and
# bash ends up with -c "source" ($0=setup.bash) → "source: filename argument
# required". Run a synced helper script by path instead (no embedded shell).
ssh -t "$HOST" \
  '~/g1_humble/humble.sh run bash /ws/src/perception/tools/robot_colcon_build.sh'

cat <<EOF

==> Deploy OK.

Na robocie (w kontenerze humble):

  # Kamera (D435i, 640x480 @ 15, aligned depth) — osobny terminal:
  ros2 launch realsense2_camera rs_launch.py \\
    camera_namespace:=camera \\
    enable_color:=true enable_depth:=true align_depth.enable:=true \\
    rgb_camera.color_profile:=640x480x15 \\
    depth_module.depth_profile:=640x480x15

  # Silnik TensorRT (po wgraniu nowego .pt; tworzy cubes_seg.engine obok):
  cd /ws
  python3 src/perception/tools/export_model.py /ws/models/cubes_seg.pt \\
    --formats engine --out /ws/models

  # Węzeł (params_robot.yaml: cubes_seg.engine na GPU,
  #        fallback cubes_seg.onnx na CPU):
  source /opt/ros/humble/setup.bash && source /ws/install/setup.bash
  export YOLO_AUTOINSTALL=false
  cd /ws
  ros2 launch g1_perception perception.launch.py config:=params_robot.yaml

  # Wymuszenie CPU albo inny model:
  ros2 launch g1_perception perception.launch.py config:=params_robot.yaml \\
    model_path:=/ws/models/cubes_seg.onnx device:=cpu
EOF
