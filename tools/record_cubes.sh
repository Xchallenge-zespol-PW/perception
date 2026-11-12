#!/usr/bin/env bash
# Nagrywa jeden krótki rosbag z kostkami i dopisuje etykietę do /bags/labels.csv
#
# Użycie (w kontenerze, kamera już uruchomiona):
#   ./record_cubes.sh <ulozenie> <swiatlo> [czas_s] [uwagi] [yaw_deg]
# Przykłady:
#   ./record_cubes.sh RGGR dzienne 15 "lekki odblask na 2 kostce"
#   ./record_cubes.sh RGGR mocne 15 "" "0,15,45,30"
#   ./record_cubes.sh RXGR mocne 15 "" "0,,45,30"
#
# <ulozenie>: kolor kostki w kolejnych miejscach od lewej, R = czerwona, G = zielona, X = brak
# [yaw_deg]:  kąty yaw w stopniach, po przecinku, tyle wartości ile liter w ułożeniu;
#             dla X zostaw pustą wartość (np. 0,,45,30)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABELS_CSV_PY="${SCRIPT_DIR}/labels_csv.py"

ARR="${1:?Podaj ulozenie, np. RGGR}"
LIGHT="${2:?Podaj swiatlo, np. dzienne / lampa / ciemno}"
DUR="${3:-15}"      # czas nagrania w SEKUNDACH
NOTE="${4:-}"
YAW="${5:-}"

if [[ ! "$ARR" =~ ^[RGX]+$ ]]; then
  echo "BLAD: ulozenie '$ARR' moze zawierac tylko litery R, G, X (np. RGGR)"; exit 1
fi
if [[ ! "$LIGHT" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "BLAD: swiatlo '$LIGHT' - uzyj liter, cyfr lub _ (np. dzienne, boczne_ostre)"; exit 1
fi
if [[ ! "$DUR" =~ ^[0-9]+$ ]] || [ "$DUR" -eq 0 ]; then
  echo "BLAD: czas '$DUR' musi byc liczba sekund, np. 15"; exit 1
fi

# Walidacja opcjonalnego yaw_deg: N pól (przecinki), X -> puste, R/G -> liczba stopni.
validate_yaw() {
  local arr="$1"
  local yaw="$2"
  local n=${#arr}

  if [ -z "$yaw" ]; then
    return 0
  fi

  local parts
  IFS=',' read -ra parts <<< "$yaw"
  if [ "${#parts[@]}" -ne "$n" ]; then
    echo "BLAD: yaw_deg ma ${#parts[@]} wartosci, oczekiwano $n (dlugosc ulozenia '$arr')"
    echo "      przyklad dla $arr: uzyj $n wartosci po przecinku; dla X zostaw puste (np. 0,,45,30)"
    exit 1
  fi

  local i ch v
  for ((i = 0; i < n; i++)); do
    ch="${arr:i:1}"
    v="${parts[$i]}"
    if [ "$ch" = "X" ]; then
      if [ -n "$v" ]; then
        echo "BLAD: miejsce $((i + 1)) to X (brak kostki) - yaw musi byc pusty, jest '$v'"
        exit 1
      fi
    else
      if [ -z "$v" ]; then
        echo "BLAD: miejsce $((i + 1)) ($ch) wymaga kata yaw w stopniach"
        exit 1
      fi
      if [[ ! "$v" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
        echo "BLAD: nieprawidlowy kat yaw '$v' na miejscu $((i + 1)) (oczekiwano liczby stopni)"
        exit 1
      fi
    fi
  done
}

validate_yaw "$ARR" "$YAW"

BAG_DIR=/bags
mkdir -p "$BAG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
NAME="kostki_${ARR}_${LIGHT}_${STAMP}"

TOPICS=(
  /camera/camera/color/image_raw
  /camera/camera/color/camera_info
  /camera/camera/aligned_depth_to_color/image_raw
  /camera/camera/aligned_depth_to_color/camera_info
  /tf
  /tf_static
)

echo "Nagrywam ${DUR}s -> ${BAG_DIR}/${NAME}"
# SIGINT po czasie = czyste zamknięcie pliku, tak jak Ctrl+C
timeout --foreground -k 10 -s INT "$DUR" ros2 bag record -s mcap -o "${BAG_DIR}/${NAME}" "${TOPICS[@]}" || true

if [ ! -d "${BAG_DIR}/${NAME}" ]; then
  echo "BLAD: nagranie nie powstalo, etykieta NIE zostala zapisana"; exit 1
fi

LABELS="${BAG_DIR}/labels.csv"
python3 "$LABELS_CSV_PY" ensure "$LABELS"
python3 "$LABELS_CSV_PY" append "$LABELS" \
  --bag "$NAME" \
  --ulozenie "$ARR" \
  --swiatlo "$LIGHT" \
  --czas-s "$DUR" \
  --uwagi "$NOTE" \
  --yaw-deg "$YAW"

echo
ros2 bag info "${BAG_DIR}/${NAME}" | grep -E "Duration|Topic:"
