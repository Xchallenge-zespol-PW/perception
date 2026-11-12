# Piłki (etap 2): interfejs `ball_node`

Dla osób, które korzystają z pozycji piłek (planer, chwytanie). Jak działa sama percepcja: README, sekcja „Węzeł piłek”.

**W skrócie:** każda czerwona piłka, którą robot zobaczył, ma stałą nazwę `ball_<N>` i pozycję swojego **środka** w ramce **`odom`**. Aktualna lista piłek to topic `/perception/balls`. TF `ball_<N>` jest wygodny do przeliczania na inne ramki.

## Uruchomienie (robot)

```bash
# terminal 1: stan robota (publikuje odom -> pelvis)
# terminal 2:
ros2 launch g1_perception balls.launch.py
```

Kamera: `/camera/d435/...` (`config/balls.yaml`). Bez TF `odom` → kamera piłki nie trafiają do mapy (w logu ostrzeżenie `No TF ...`).

## Wyjścia

| Wyjście | Typ | Opis |
|---------|-----|------|
| `/perception/balls` | `vision_msgs/Detection3DArray` | **Aktualna lista piłek**, 5 Hz. `header.frame_id = odom`, stempel = teraz. Dla każdej piłki: `id` = nazwa (`ball_3`), `bbox.center.position` = środek w m, `bbox.size.x/y/z` = zmierzona średnica (ok. 0,066 m), `results[0].hypothesis.class_id = red ball`. Orientacja: tożsamość (bez znaczenia). Pusta lista = brak piłek |
| TF `ball_<N>` | `odom → ball_<N>` | To samo co wyżej, 5 Hz, stempel = teraz, rotacja = tożsamość |
| `/perception/balls/summary` | `std_msgs/String` | To samo jako JSON, 5 Hz: `[{"name": "ball_1", "x": 1.486, "y": -0.828, "z": 0.433, "diameter_m": 0.069}, ...]` (w `odom`); do podglądu i prostych skryptów |
| `/ball_node/reset` | `std_srvs/Trigger` | Czyści mapę przed nowym przejazdem; numeracja od `ball_1` |
| TF `ball_seen_<K>`, `/perception/balls/seen` | TF / `geometry_msgs/PoseArray` | Tylko diagnostyka: piłki z ostatniej klatki, `K` = kolejność od najbliższej, **nie** stała nazwa. Nie używać do planowania |
| `/perception/balls/debug_image` | `sensor_msgs/Image` | Podgląd (otwierać tylko na chwilę, spowalnia węzeł) |

## Cykl życia piłki

1. **Pojawienie się**: piłka widoczna w 5 klatkach (ok. 0,3 s) dostaje kolejną nazwę `ball_<N>` i od tej chwili jest w `/perception/balls`. Pojedyncze fałszywe wykrycia nie dostają nazwy.
2. **Poza kadrem lub zasłonięta** (ręka, słupek regału): zostaje z ostatnią pozycją. Pozycja to mediana ostatnich 30 obserwacji.
3. **Zabrana**: gdy jej miejsce jest w kadrze, głębia pokazuje, że nic tam nie ma, i tak jest przez ≥ 1 s, piłka **znika** z `/perception/balls` i TF `ball_<N>` przestaje być publikowany.
4. **Odłożona z powrotem**: dostaje **nową** nazwę. Nazwy nigdy się nie powtarzają w jednym przejeździe (do `reset`).

## Ramka i współrzędne

- **`odom`**: ramka nieruchoma względem areny w czasie przejazdu, a początek jest tam, gdzie robot zaczął. `z` to wysokość w `odom`, nie wysokość nad podłogą.
- Pozycja to **środek** piłki. Chwyt planuje się względem środka.
- `odom` dryfuje przy chodzeniu. Jak bardzo, pokazuje `match_mm` w logu `ball stats` węzła. Pozycja piłki jest uaktualniana przy każdym jej zobaczeniu, więc piłka widoczna z bliska ma pozycję zgodną z aktualnym `odom`.

## Jak czytać pozycję

**Lista piłek** ze źródła prawdy, czyli `/perception/balls`. **Przeliczenie** na inną ramkę (np. `pelvis`) przez TF, z kontrolą wieku.

Uwaga: `lookup_transform(..., Time())` zwraca **ostatni znany** TF, także dla piłki zabranej przed minutą. Dlatego sprawdzaj obecność piłki w `/perception/balls` albo wiek stempla.

Sprawdzone z węzłem na bagu (`ball_in` zwraca środki w ramce kamery, `None` dla nieznanej nazwy).

```python
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray


class BallReader(Node):
    def __init__(self):
        super().__init__('ball_reader')
        self.balls = {}  # nazwa -> (x, y, z) w odom
        self.create_subscription(
            Detection3DArray, '/perception/balls', self.on_balls, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def on_balls(self, msg):
        self.balls = {
            d.id: (
                d.bbox.center.position.x,
                d.bbox.center.position.y,
                d.bbox.center.position.z,
            )
            for d in msg.detections
        }

    def ball_in(self, name, frame='pelvis', max_age_s=1.0):
        """Środek piłki w ``frame`` albo None (brak / zabrana / stary TF)."""
        if name not in self.balls:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(frame, name, Time())
        except TransformException:
            return None
        age = self.get_clock().now() - Time.from_msg(tf.header.stamp)
        if age.nanoseconds * 1e-9 > max_age_s:
            return None
        t = tf.transform.translation
        return t.x, t.y, t.z
```

Z terminala:

```bash
ros2 topic echo /perception/balls/summary                # lista piłek (JSON)
ros2 run tf2_ros tf2_echo pelvis ball_1                  # jedna piłka względem miednicy
ros2 service call /ball_node/reset std_srvs/srv/Trigger  # nowy przejazd
```

W RViz: Fixed Frame `odom`, wyświetlanie TF.

## Czego węzeł nie robi (na razie)

- **Nie przypisuje piłki do półki ani do miejsca na regale.** Wysokość jest w `z` (`odom`), a półki są na 60 i 100 cm nad podłogą.
- **Nie koryguje dryfu `odom`**, np. z AprilTagów 9–14 na końcach regałów. Do zrobienia, jeśli `match_mm` na robocie okaże się duże.
- **Nie rozróżnia kolorów** (są tylko czerwone piłki) i nie podaje orientacji.
- **Nie podaje pewności ani wieku obserwacji** w wiadomości (`score` = 1). Wiek ostatniego zobaczenia widać tylko w TF `ball_seen_<K>`.

## Do ustalenia z planerem

- Czy wystarczy środek w `odom`, czy potrzebna ramka podłogi lub areny (np. z AprilTagów)?
- Czy potrzebny numer półki albo miejsca na regale?
- Czy potrzebny wiek ostatniego zobaczenia piłki albo pewność?
- Co ma się stać z nazwami po odłożeniu piłki? Teraz piłka dostaje nową nazwę.
