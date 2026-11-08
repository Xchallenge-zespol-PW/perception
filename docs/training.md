# Dane i modele: seg + pose

Przepływ od nagrania do modeli na robocie dla obu segmenterów: `yolo_seg` (maska całej kostki) i `yolo_pose` (4 narożniki górnej ściany, konwencje w [pose_pnp.md](pose_pnp.md)). Jedno etykietowanie daje oba datasety.

```text
record_cubes.sh → extract_frames.py → CVAT (poligon + 4 narożniki)
  → cvat_to_yolo.py → cubes_seg/ + cubes_pose/ → preview_pose_labels.py
  → train_yolo.py (Colab GPU) → export_model.py (.pt + .onnx)
  → deploy_robot.sh → export_model.py --formats engine (na Jetsonie) → compare_backends.py → bench_models.py
```

Laptop nie ma CUDA (AMD), więc trening idzie w Colabie; na laptopie konwersja, podgląd, eksport `.onnx` i bench na CPU.

## 0. Przed treningiem: czy YOLO26 ruszy na Jetsonie

W kontenerze humble na robocie, po uruchomieniu GPU:

```bash
YOLO_AUTOINSTALL=false python3 /ws/src/perception/tools/jetson_gpu_check.py \
  --export yolo26n-pose.pt --workdir /ws/models/check
```

Wszystkie linie `PASS` = środowisko gotowe, a eksport YOLO26 do TensorRT działa. Gdy `engine export` daje `FAIL`, powtórz z `--export yolo11n-pose.pt` i trenuj na wagach `yolo11*` (`--model yolo11s-pose.pt`).

## 1. Nagrania

Do oceny `pnp` vs `geometric` potrzebny jest **znany yaw** każdej kostki (mata z kątami):

```bash
./tools/record_cubes.sh RGGR mocne 15 "" "0,15,45,30"
python3 tools/extract_frames.py
```

Różne światło (`dzienne`, `mocne`, `niskie`, `ciemnenoise`) i ułożenia; kostki także stykające się i obrócone o 30–45°. Klatki trafiają do `data/frames/<bag>/<n>_color.png`.

## 2. Etykietowanie w CVAT

Projekt z etykietami **`red`** i **`green`** (inne nazwy mapuje `config/labels.yaml`, np. `czerwona`). Zadanie: wgraj klatki **z katalogami nagrań** (zip `data/frames/`), bo podział train/val idzie po nagraniu = pierwszym katalogu nazwy obrazu.

Na każdą kostkę:

| Kształt | Co zaznaczyć |
|---------|--------------|
| **Polygon** | obrys całej widocznej kostki (góra + boki) → maska seg i prostokąt pose |
| **Points** (4 punkty) albo **Skeleton** z 4 punktami | narożniki **górnej ściany**, w dowolnej kolejności |

- Kolejność punktów nie ma znaczenia: konwerter układa je w `c0..c3` (wizualnie zgodnie z ruchem wskazówek zegara, `c0` = najniższy, przy różnicy `v` < 3 px ten bardziej z prawej).
- Punkty przypisują się do poligonu przez wspólny **group** (Ctrl+G) albo, bez grupy, do najmniejszego poligonu, w którym leży ich środek. Przy stykających się kostkach grupuj.
- Zasłonięty narożnik, którego położenie widać: zostaw punkt i ustaw `occluded` (widoczność 1). Narożnik niemożliwy do wskazania: kostka bez punktów (skeleton: `outside`) — w pose dostanie same zera.
- Kostka, której górnej ściany nie widać: tylko poligon.

Eksport: **CVAT for images 1.1** (XML).

## 3. Konwersja i kontrola

```bash
python3 tools/cvat_to_yolo.py annotations.xml --images-root data/frames
# albo ze stałymi nagraniami walidacyjnymi:
python3 tools/cvat_to_yolo.py annotations.xml --images-root data/frames \
  --val-groups kostki_RGGR_niskie_20261015_184332 kostki_RGRG_ciemnenoise_20261015_185122
python3 tools/preview_pose_labels.py datasets/cubes_pose --split train --limit 40
```

- Wynik: `datasets/cubes_seg/` i `datasets/cubes_pose/` (`images/`, `labels/`, `data.yaml` z `kpt_shape: [4, 3]`, `flip_idx: [0, 3, 2, 1]`). Istniejący katalog nie jest nadpisywany bez `--overwrite` (stary dataset z Roboflow przenieś albo użyj `--seg-name`).
- Ostrzeżenia `WARN` (punkty poza poligonem, nieznana etykieta, brak obrazu) przejrzyj przed treningiem.
- `preview_pose_labels.py` zwraca kod 1, gdy któraś etykieta łamie kolejność `c0..c3` (np. dataset etykietowany poza CVAT), i zapisuje podgląd z numerami narożników do `<dataset>/preview_<split>/`.

## 4. Trening (Colab GPU)

```bash
pip install -q -c deploy/constraints.txt ultralytics
python3 tools/train_yolo.py --task seg --device 0
python3 tools/train_yolo.py --task pose --device 0
# większy model, gdy bench na robocie pozwala:
python3 tools/train_yolo.py --task pose --model yolo26s-pose.pt --device 0
```

- Domyślnie `yolo26n-seg.pt` / `yolo26n-pose.pt`, 100 epok, `imgsz=640`, `hsv_h=0`; pose z `fliplr=0` (odbicie łamie regułę `c0`).
- `ultralytics==8.4.191` z `deploy/constraints.txt` — ta sama wersja w Colabie, na laptopie i na robocie.
- Wagi: `runs/seg/cubes/weights/best.pt`, `runs/pose/cubes_pose/weights/best.pt`.

## 5. Eksport i wdrożenie

Na laptopie (albo w Colabie):

```bash
python3 tools/export_model.py runs/seg/cubes/weights/best.pt --name cubes_seg
python3 tools/export_model.py runs/pose/cubes_pose/weights/best.pt --name cubes_pose
./tools/deploy_robot.sh models/cubes_seg.pt models/cubes_seg.onnx \
  models/cubes_pose.pt models/cubes_pose.onnx
```

Na robocie (w kontenerze), silniki TensorRT z wgranych `.pt` — `.onnx` (fallback CPU) zostaje nietknięty:

```bash
cd /ws
python3 src/perception/tools/export_model.py /ws/models/cubes_seg.pt --formats engine --out /ws/models
python3 src/perception/tools/export_model.py /ws/models/cubes_pose.pt --formats engine --out /ws/models
# silnik (GPU, FP16) vs .onnx (CPU) na tych samych klatkach — ma być OK:
python3 src/perception/tools/compare_backends.py /ws/models/cubes_pose.engine /ws/models/cubes_pose.onnx --frames /ws/data/frames
```

`params_robot.yaml` ładuje `/ws/models/cubes_seg.engine` na GPU z fallbackiem `/ws/models/cubes_seg.onnx` na CPU ([robot.md](robot.md)).

## 6. Bench i wybór rozmiaru

```bash
# robot: GPU (.engine) i fallback CPU (.onnx) na prawdziwych klatkach
python3 src/perception/tools/bench_models.py /ws/models/cubes_pose.engine \
  /ws/models/cubes_pose.onnx --device 0 --frames /ws/data/frames --csv /ws/results/bench_robot.csv
# laptop: CPU
python3 tools/bench_models.py models/cubes_pose.onnx models/cubes_seg.onnx --device cpu
```

Wybierz największy rozmiar (`n` / `s` / `m`), który na GPU robota mieści się z zapasem w 15 kl/s kamery (~66 ms), a na fallbacku CPU wciąż daje sensowne głosowanie. Wynik wpisz do tabeli w [robot.md](robot.md).
