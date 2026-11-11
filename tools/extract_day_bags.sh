#!/usr/bin/env bash
# Wyciąga klatki z bagów nagranych jednego dnia (domyślnie dziś) do osobnego
# datasetu: <out-dir>/frames/<bag>/ + <out-dir>/index.csv.
#
# Użycie (z /ws):
#   ./tools/extract_day_bags.sh                     # dzisiejsze kostki -> data/kostki_<data>
#   ./tools/extract_day_bags.sh --date 20261107 --every 10
#   ./tools/extract_day_bags.sh --prefix pilki      # piłki, etykiety z labels_pilki.csv

set -euo pipefail

DATE=$(date +%Y%m%d)
PREFIX=kostki
BAGS_DIR=/bags
LABELS=
OUT_DIR=
EVERY=15
MAX_DT_MS=20

usage() {
  echo "Użycie: $0 [--date YYYYMMDD] [--prefix kostki|pilki] [--bags-dir /bags]"
  echo "          [--labels CSV] [--out-dir data/<prefix>_<date>] [--every 15] [--max-dt-ms 20]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --date) DATE="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --bags-dir) BAGS_DIR="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --every) EVERY="$2"; shift 2 ;;
    --max-dt-ms) MAX_DT_MS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Nieznany argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$DATE" =~ ^[0-9]{8}$ ]]; then
  echo "BLAD: data '$DATE' musi miec format YYYYMMDD, np. 20261111" >&2; exit 2
fi
case "$PREFIX" in
  kostki) LABELS="${LABELS:-$BAGS_DIR/labels.csv}" ;;
  pilki) LABELS="${LABELS:-$BAGS_DIR/labels_pilki.csv}" ;;
  *) echo "BLAD: prefix '$PREFIX' - dozwolone kostki albo pilki" >&2; exit 2 ;;
esac
if [[ ! "$EVERY" =~ ^[0-9]+$ ]] || [ "$EVERY" -eq 0 ]; then
  echo "BLAD: --every '$EVERY' musi byc liczba >= 1" >&2; exit 2
fi
if [[ ! -f "$LABELS" ]]; then
  echo "BLAD: brak pliku etykiet $LABELS" >&2; exit 1
fi
OUT_DIR="${OUT_DIR:-data/${PREFIX}_${DATE}}"
BAGS_DIR=$(realpath "$BAGS_DIR")
LABELS=$(realpath "$LABELS")

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Mini-CSV tylko z bagami z tego dnia -> jedno wywołanie extract_frames.
head -1 "$LABELS" > "$TMP/labels.csv"
grep -E "^${PREFIX}_[^,]*_${DATE}_[0-9]{6}," "$LABELS" >> "$TMP/labels.csv" || true

N=0
FAIL=0
while IFS=, read -r BAG _; do
  if [[ -d "$BAGS_DIR/$BAG" ]]; then
    N=$((N + 1))
  else
    echo "BLAD: $BAG jest w $LABELS, ale brak katalogu $BAGS_DIR/$BAG" >&2
    FAIL=1
  fi
done < <(tail -n +2 "$TMP/labels.csv")

# Bagi z tego dnia na dysku, ale bez etykiety, extract_frames by pominął po cichu.
for DIR in "$BAGS_DIR/${PREFIX}_"*"_${DATE}_"*; do
  [[ -d "$DIR" ]] || continue
  BAG=$(basename "$DIR")
  if ! grep -q "^${BAG}," "$TMP/labels.csv"; then
    echo "Uwaga: $BAG nie ma wiersza w $LABELS - pomijam" >&2
    FAIL=1
  fi
done

if [[ "$N" -eq 0 ]]; then
  echo "BLAD: brak bagow ${PREFIX}_*_${DATE}_* w $LABELS" >&2; exit 1
fi

echo "==> ${PREFIX} ${DATE}: ${N} bagow, co ${EVERY}. klatka -> ${OUT_DIR}"
mkdir -p "$OUT_DIR"
python3 tools/extract_frames.py \
  --labels "$TMP/labels.csv" \
  --bags-dir "$BAGS_DIR" \
  --out-dir "$OUT_DIR" \
  --every "$EVERY" \
  --max-dt-ms "$MAX_DT_MS"

if [[ "$FAIL" -ne 0 ]]; then
  echo "Uwaga: czesc bagow pominieta (brak w etykietach lub na dysku)." >&2
  exit 1
fi
echo "Gotowe: $OUT_DIR/frames/ + $OUT_DIR/index.csv"
