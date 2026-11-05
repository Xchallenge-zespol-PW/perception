# Poza kostki z YOLO-pose + PnP: konwencje

Specyfikacja wspólna dla etykiet, modelu PnP i kodu (`segmenter: yolo_pose`, `pose_estimator: pnp`). Każda zmiana tutaj wymaga zmiany etykiet, testów i kodu jednocześnie.

Oznaczenia: `S = cube_size_m` (domyślnie 0,05 m), obraz w pikselach `(u, v)` z `u` w prawo i `v` w dół, ramka kamery optyczna (x w prawo, y w dół, z do przodu).

## Punkty kluczowe

Model wykrywa 4 narożniki **górnej ściany** kostki (tej równoległej do stołu, widocznej z góry). Narożników dolnych i bocznych nie etykietujemy.

| Punkt | Reguła |
|-------|--------|
| `c0` | narożnik o największym `v` (najniżej na obrazie = najbliżej kamery). Jeśli dwa narożniki różnią się w `v` o mniej niż 3 px, `c0` to ten o większym `u` (prawy) |
| `c1`, `c2`, `c3` | kolejne narożniki **wizualnie zgodnie z ruchem wskazówek zegara** na obrazie, licząc od `c0` |

Zgodnie z ruchem wskazówek zegara przy `v` w dół oznacza dodatnie pole ze wzoru shoelace: `Σ (u_i·v_{i+1} − u_{i+1}·v_i) > 0`.

**Co z tej kolejności naprawdę wynika.** Górna ściana jest kwadratem, a yaw podajemy modulo 90°. Cykliczne przesunięcie `c0..c3` (np. `c1, c2, c3, c0`) daje tę samą pozę. Dla poprawności PnP liczy się tylko **kierunek obiegu**; reguła wyboru `c0` służy spójności etykiet, żeby model nie uśredniał dwóch narożników.

Dlatego kod przed PnP **nie ufa kolejności z modelu**: sortuje 4 punkty kątowo wokół ich środka do obiegu zgodnego z ruchem wskazówek zegara i wybiera `c0` regułą z tabeli. Zamienione lub odbite indeksy z modelu nie psują wtedy orientacji.

### Widoczność

Trzecia wartość punktu w etykiecie (Ultralytics pose):

| Wartość | Znaczenie |
|---------|-----------|
| `2` | narożnik widoczny |
| `1` | zasłonięty, ale położenie da się wskazać (np. przez sąsiednią kostkę) |
| `0` | poza obrazem / nie da się wskazać; współrzędne `0 0` |

Kostki, których górnej ściany nie widać w ogóle, etykietujemy tylko prostokątem z wszystkimi punktami `0 0 0`.

## Format datasetu

Ultralytics pose, jeden plik `.txt` na obraz, jedna linia na kostkę:

```text
<class> <cx> <cy> <w> <h> <u0> <v0> <vis0> <u1> <v1> <vis1> <u2> <v2> <vis2> <u3> <v3> <vis3>
```

Wszystkie współrzędne znormalizowane do [0, 1]; prostokąt obejmuje **całą widoczną kostkę** (z bokami), nie tylko górną ścianę.

```yaml
# data.yaml
kpt_shape: [4, 3]
flip_idx: [0, 3, 2, 1]
names: ['red', 'green']   # inne nazwy mapuje core.labels (labels.yaml)
```

- `flip_idx` odwraca obieg po odbiciu poziomym, ale nie przywraca reguły wyboru `c0`. Trening z **`fliplr=0.0`**, żeby etykiety po augmentacji spełniały tę specyfikację.
- **`hsv_h=0.0`**: kolor to klasa, przesunięcie odcienia zamieniałoby czerwień w zieleń.

## Model PnP

Górna ściana jako kwadrat w płaszczyźnie `z = 0` własnej ramki, w kolejności wymaganej przez `cv2.SOLVEPNP_IPPE_SQUARE`:

| Indeks | Punkt modelu | Punkt obrazu |
|--------|--------------|--------------|
| 0 | `[-S/2,  S/2, 0]` | `c0` |
| 1 | `[ S/2,  S/2, 0]` | `c1` |
| 2 | `[ S/2, -S/2, 0]` | `c2` |
| 3 | `[-S/2, -S/2, 0]` | `c3` |

Oglądany od strony `+z` modelu ten obieg jest zgodny z ruchem wskazówek zegara na obrazie. Mapowanie `c_i → i` jest więc bezpośrednie, a oś `+z` modelu wychodzi z górnej ściany w górę, w stronę kamery (w ramce kamery `z` normalnej < 0). Sprawdzone rzutem syntetycznym dla pochyleń kamery 30–70° i dowolnego yaw. Test mapowania należy do PR 3 (`test/test_pnp.py`).

`solvePnPGeneric` z IPPE zwraca dwa rozwiązania. Wybieramy to, którego oś `+z` jest bliższa normalnej stołu (z `table_plane_around`, skierowanej do kamery); błąd reprojekcji rozstrzyga tylko, gdy oba są równie zgodne z normalną.

## Orientacja

Rotacja z PnP jest rzutowana na płaszczyznę stołu, tak samo jak w `geometric`:

- oś `+Z` pozy = normalna stołu skierowana do kamery (w górę od stołu),
- oś `+X` = rzut osi `+x` modelu PnP na płaszczyznę stołu,
- `yaw_deg` = kąt `+X` w bazie `plane_basis(n)` (`core.geometry`), modulo 90°, w [0, 90),
- kwaternion `(x, y, z, w)` z `quat_from_plane_yaw(n, yaw_deg)`.

Pochylenie i przechył z PnP są odrzucane, bo kostka leży płasko. Liczba `yaw_deg` zależy od bazy płaszczyzny; kontraktem dla odbiorców jest **kwaternion**.

## Pozycja i środek bryły

Pozycja pochodzi **z głębi**, nie z translacji PnP:

1. Punkty 3D z głębi wewnątrz czworokąta `c0..c3` (zawężonego erozją o kilka px, żeby odciąć krawędzie i flying pixels).
2. Mediana tych punktów = środek górnej ściany `p_top`.
3. Środek bryły: `p = p_top − (S/2)·n`, gdzie `n` to normalna stołu skierowana do kamery (przesunięcie w stronę stołu).

To ten sam środek, który daje `geometric` (`cube_pose_on_table`: środek na stole `+ (S/2)·n`), więc obie metody można porównywać bezpośrednio.

Translacja PnP służy tylko do kontroli spójności z głębią (`pnp_max_depth_diff_m`). Spodziewany szum przy D435i 640×480 (`fx ≈ 615`):

| Odległość | Bok kostki na obrazie | Błąd odległości z PnP przy 1 px |
|-----------|-----------------------|---------------------------------|
| 0,6 m | ~51 px | ~1,2 cm |
| 1,0 m | ~31 px | ~3 cm |

Próg rozbieżności musi to uwzględniać (stały próg 3 cm przy 1 m odrzuci znaczną część poprawnych estymacji).

## Maska z czworokąta

`quad_mask` wypełnia czworokąt `c0..c3`, więc obejmuje **tylko górną ścianę**. Do wykluczenia obiektu przy dopasowaniu płaszczyzny stołu (`table_plane_around`) trzeba używać prostokąta detekcji, inaczej boki kostki trafią do pierścienia stołu. `/perception/mask` przy `yolo_pose` pokazuje górne ściany.

## Ramki TF

| Element | Ustalenie |
|---------|-----------|
| Rodzic | `target_frame` |
| Dziecko | `{tf_child_prefix}{id}`, gdzie `id` = `detections[i].id` z `/perception/cubes` (w `rois_mode=none` numer miejsca `1`…`N`); domyślnie `cube_1` … `cube_N` |
| Stempel | stempel obrazu kolorowego klatki źródłowej (ten sam co w `/perception/cubes`) |
| Wartość | ta sama poza co `results[0].pose.pose` (środek bryły, kwaternion jak wyżej) |

Ramka `cube_N` **nie** jest publikowana, gdy:

- etykieta to `uncertain` albo `empty`, albo kolor bez pozy,
- walidacja miejsc nie przeszła (`assignment_ok = False`; `id` jest wtedy id trackera, nie numerem miejsca),
- nie udał się lookup TF kamera → `target_frame` (poza zostałaby w ramce kamery, pod złym rodzicem).

Odbiorcy powinni sprawdzać stempel transformacji: ramka kostki, która zniknęła, pozostaje w ich buforze TF jako ostatnia znana poza.
