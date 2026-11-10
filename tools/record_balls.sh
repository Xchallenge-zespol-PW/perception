#!/usr/bin/env bash
# Nagrywa jeden krótki rosbag z piłkami (etap 2) i dopisuje etykietę do /bags/labels_pilki.csv
#
# Użycie (w kontenerze, kamera już uruchomiona):
#   ./record_balls.sh <liczba_pilek> <polka> <swiatlo> [czas_s] [uwagi]
# Przykłady:
#   ./record_balls.sh 4 h60 lampa
#   ./record_balls.sh 2 h60_100 dzienne 15 "jedna pilka przy krawedzi"
#   ./record_balls.sh 0 stol sufit_lampa
#
# <liczba_pilek>: 0-4
# <polka>:        h60, h100, h60_100 (obie półki) albo stol
# [czas_s]:       domyślnie 10
#
# Katalog bagów można nadpisać zmienną BAG_DIR (domyślnie /bags).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABELS_CSV_PY="${SCRIPT_DIR}/labels_csv.py"

COUNT="${1:?Podaj liczbe pilek 0-4}"
SHELF="${2:?Podaj polke: h60 / h100 / h60_100 / stol}"
LIGHT="${3:?Podaj swiatlo, np. dzienne / lampa / ciemno}"
DUR="${4:-10}"      # czas nagrania w SEKUNDACH
NOTE="${5:-}"

if [[ ! "$COUNT" =~ ^[0-4]$ ]]; then
  echo "BLAD: liczba pilek '$COUNT' musi byc od 0 do 4"; exit 1
fi
case "$SHELF" in
  h60|h100|h60_100|stol) ;;
  *) echo "BLAD: polka '$SHELF' - dozwolone: h60, h100, h60_100, stol"; exit 1 ;;
esac
if [[ ! "$LIGHT" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "BLAD: swiatlo '$LIGHT' - uzyj liter, cyfr lub _ (np. dzienne, boczne_ostre)"; exit 1
fi
if [[ ! "$DUR" =~ ^[0-9]+$ ]] || [ "$DUR" -eq 0 ]; then
  echo "BLAD: czas '$DUR' musi byc liczba sekund, np. 10"; exit 1
fi

BAG_DIR="${BAG_DIR:-/bags}"
mkdir -p "$BAG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
NAME="pilki_${COUNT}_${SHELF}_${LIGHT}_${STAMP}"

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

LABELS="${BAG_DIR}/labels_pilki.csv"
python3 "$LABELS_CSV_PY" ensure-balls "$LABELS"
python3 "$LABELS_CSV_PY" append-balls "$LABELS" \
  --bag "$NAME" \
  --liczba-pilek "$COUNT" \
  --polka "$SHELF" \
  --swiatlo "$LIGHT" \
  --czas-s "$DUR" \
  --uwagi="$NOTE"

echo
ros2 bag info "${BAG_DIR}/${NAME}" | grep -E "Duration|Topic:"
