#!/usr/bin/env bash
# Nagrywa jeden krótki rosbag z kostkami i dopisuje etykietę do /bags/labels.csv
#
# Użycie (w kontenerze, kamera już uruchomiona):
#   ./record_cubes.sh <ulozenie> <swiatlo> [czas_s] [uwagi]
# Przykład:
#   ./record_cubes.sh RGGR dzienne 15 "lekki odblask na 2 kostce"
#
# <ulozenie>: kolor kostki w kolejnych miejscach od lewej, R = czerwona, G = zielona, X = brak

set -e

ARR="${1:?Podaj ulozenie, np. RGGR}"
LIGHT="${2:?Podaj swiatlo, np. dzienne / lampa / ciemno}"
DUR="${3:-15}"      # czas nagrania w SEKUNDACH
NOTE="${4:-}"

if [[ ! "$ARR" =~ ^[RGX]+$ ]]; then
  echo "BLAD: ulozenie '$ARR' moze zawierac tylko litery R, G, X (np. RGGR)"; exit 1
fi
if [[ ! "$LIGHT" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "BLAD: swiatlo '$LIGHT' - uzyj liter, cyfr lub _ (np. dzienne, boczne_ostre)"; exit 1
fi
if [[ ! "$DUR" =~ ^[0-9]+$ ]] || [ "$DUR" -eq 0 ]; then
  echo "BLAD: czas '$DUR' musi byc liczba sekund, np. 15"; exit 1
fi

BAG_DIR=/bags
mkdir -p "$BAG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
NAME="kostki_${ARR}_${LIGHT}_${STAMP}"

TOPICS=(
  /camera/camera/color/image_raw
  /camera/camera/color/camera_info
  /camera/camera/aligned_depth_to_color/image_raw
  /camera/camera/aligned_depth_to_color/camera_info
  /tf_static
)

echo "Nagrywam ${DUR}s -> ${BAG_DIR}/${NAME}"
# SIGINT po czasie = czyste zamknięcie pliku, tak jak Ctrl+C
timeout --foreground -k 10 -s INT "$DUR" ros2 bag record -s mcap -o "${BAG_DIR}/${NAME}" "${TOPICS[@]}" || true

if [ ! -d "${BAG_DIR}/${NAME}" ]; then
  echo "BLAD: nagranie nie powstalo, etykieta NIE zostala zapisana"; exit 1
fi

LABELS="${BAG_DIR}/labels.csv"
[ -f "$LABELS" ] || echo "bag,ulozenie,swiatlo,czas_s,uwagi" > "$LABELS"
echo "${NAME},${ARR},${LIGHT},${DUR},\"${NOTE}\"" >> "$LABELS"

echo
ros2 bag info "${BAG_DIR}/${NAME}" | grep -E "Duration|Topic:"
