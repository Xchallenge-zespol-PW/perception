"""Load ROI and HSV calibration YAML into dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Roi:
    """
    Axis-aligned rectangle for one cube slot on the color image.

    Coordinates are inclusive-exclusive pixel bounds: [x0, x1) x [y0, y1).
    """

    slot: int
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class HsvRange:
    """
    Inclusive HSV bounds in OpenCV units.

    Hue is in ``[0, 180]``, saturation and value in ``[0, 255]``.
    """

    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int


@dataclass(frozen=True)
class HsvConfig:
    """
    HSV thresholds for red (two hue wraps) and green cubes.

    Red uses ``red_low`` (near hue 0) and ``red_high`` (near hue 180).
    """

    red_low: HsvRange
    red_high: HsvRange
    green: HsvRange


def default_config_dir() -> Path:
    """
    Return the package ``config/`` directory.

    Resolves to ``src/g1_perception/config`` next to the ament package root.
    """
    return Path(__file__).resolve().parents[2] / 'config'


def resolve_config_path(name: str, path: Path | None = None) -> Path:
    """
    Resolve a config file path, defaulting to :func:`default_config_dir`.

    ``name`` is a file name such as ``rois.yaml`` or ``hsv.yaml``.
    """
    if path is not None:
        return Path(path)
    return default_config_dir() / name


def load_rois(path: Path | None = None) -> list[Roi]:
    """
    Load ROI rectangles from ``rois.yaml``.

    Expected YAML::

        rois:
          - slot: 0
            x0: 10
            y0: 20
            x1: 100
            y1: 120
    """
    cfg_path = resolve_config_path('rois.yaml', path)
    with cfg_path.open(encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    raw = data.get('rois', data if isinstance(data, list) else [])
    if not isinstance(raw, list):
        raise ValueError(f'Invalid rois.yaml format in {cfg_path}')

    rois: list[Roi] = []
    for item in raw:
        rois.append(
            Roi(
                slot=int(item['slot']),
                x0=int(item['x0']),
                y0=int(item['y0']),
                x1=int(item['x1']),
                y1=int(item['y1']),
            )
        )
    return sorted(rois, key=lambda r: r.slot)


def load_hsv(path: Path | None = None) -> HsvConfig:
    """
    Load red/green HSV ranges from ``hsv.yaml``.

    Expected keys: ``red_low``, ``red_high``, ``green``, each with
    ``h_min``, ``h_max``, ``s_min``, ``s_max``, ``v_min``, ``v_max``.
    """
    cfg_path = resolve_config_path('hsv.yaml', path)
    with cfg_path.open(encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    def _range(key: str) -> HsvRange:
        try:
            block = data[key]
        except KeyError as exc:
            raise KeyError(f'Missing {key!r} in {cfg_path}') from exc
        return HsvRange(
            h_min=int(block['h_min']),
            h_max=int(block['h_max']),
            s_min=int(block['s_min']),
            s_max=int(block['s_max']),
            v_min=int(block['v_min']),
            v_max=int(block['v_max']),
        )

    return HsvConfig(
        red_low=_range('red_low'),
        red_high=_range('red_high'),
        green=_range('green'),
    )


def save_rois(rois: list[Roi], path: Path | None = None) -> Path:
    """Write ROI list to ``rois.yaml`` and return the path written."""
    cfg_path = resolve_config_path('rois.yaml', path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'rois': [
            {
                'slot': r.slot,
                'x0': r.x0,
                'y0': r.y0,
                'x1': r.x1,
                'y1': r.y1,
            }
            for r in rois
        ]
    }
    with cfg_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return cfg_path


def save_hsv(cfg: HsvConfig, path: Path | None = None) -> Path:
    """Write HSV thresholds to ``hsv.yaml`` and return the path written."""
    cfg_path = resolve_config_path('hsv.yaml', path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    def _dump(r: HsvRange) -> dict[str, int]:
        return {
            'h_min': r.h_min,
            'h_max': r.h_max,
            's_min': r.s_min,
            's_max': r.s_max,
            'v_min': r.v_min,
            'v_max': r.v_max,
        }

    payload = {
        'red_low': _dump(cfg.red_low),
        'red_high': _dump(cfg.red_high),
        'green': _dump(cfg.green),
    }
    with cfg_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return cfg_path
