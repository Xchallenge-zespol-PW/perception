Projekt: percepcja kostek dla robota Unitree G1 EDU (zawody). Repo: pakiet ROS 2 Humble (ament_python) w src/g1_perception, Python 3.10, uruchamiane w devcontainerze. numpy<2 (cv_bridge). Kamera Intel RealSense, obraz kolorowy + głębia wyrównana do koloru (uint16, mm) + camera_info.

Zadanie: dla 4 znanych miejsc na stole wykryć kostkę (czerwona/zielona/brak, sześcian 5 cm) i wyznaczyć jej pozę (pozycja 3D + orientacja). Architektura: 3 wymienne etapy (segmentacja -> kolor -> poza), wybierane z konfiguracji. Logika w g1_perception/core, segmenters/, pose/, pipeline.py — BEZ importów ROS. Węzeł ROS 2 to cienka warstwa na końcu. Narzędzia offline w tools/. Testy pytest w src/g1_perception/test.

Nagrania: rosbagi MCAP w /bags, etykiety w /bags/labels.csv (kolumny: bag, ulozenie np. RGGR, swiatlo, czas_s, uwagi). Topici: /camera/camera/color/image_raw, /camera/camera/aligned_depth_to_color/image_raw, /camera/camera/color/camera_info.

Zasady: małe, czytelne moduły, type hints, docstringi, bez zależności od ROS w logice, wszystko konfigurowalne przez YAML, każda funkcja logiki z testem.