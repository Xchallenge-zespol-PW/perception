#!/usr/bin/env python3
"""
Helpers for label CSVs written by the recording scripts.

``/bags/labels.csv`` is used by record_cubes.sh, ``/bags/labels_pilki.csv`` by
record_balls.sh.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

FIELDNAMES = ['bag', 'ulozenie', 'swiatlo', 'czas_s', 'uwagi', 'yaw_deg']
BALL_FIELDNAMES = ['bag', 'liczba_pilek', 'polka', 'swiatlo', 'czas_s', 'uwagi']


def ensure_labels_csv(path: Path) -> None:
    """
    Create labels.csv or add a missing ``yaw_deg`` column.

    Existing rows get an empty ``yaw_deg`` value when the column is added.
    """
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        return

    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            with path.open('w', newline='', encoding='utf-8') as out:
                csv.DictWriter(out, fieldnames=FIELDNAMES).writeheader()
            return
        rows = list(reader)
        existing = list(reader.fieldnames)

    if 'yaw_deg' in existing:
        return

    print(
        f'DOPISANO kolumne yaw_deg do {path} '
        f'(puste wartosci dla istniejacych wierszy)',
        file=sys.stderr,
    )
    migrated = [{name: row.get(name, '') for name in FIELDNAMES} for row in rows]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(migrated)


def append_label(
    path: Path,
    *,
    bag: str,
    ulozenie: str,
    swiatlo: str,
    czas_s: str,
    uwagi: str,
    yaw_deg: str,
) -> None:
    """Append one label row to ``path`` (file must already exist)."""
    row = {
        'bag': bag,
        'ulozenie': ulozenie,
        'swiatlo': swiatlo,
        'czas_s': czas_s,
        'uwagi': uwagi,
        'yaw_deg': yaw_deg,
    }
    with path.open('a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def ensure_ball_labels_csv(path: Path) -> None:
    """Create labels_pilki.csv with a header if it is missing or empty."""
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=BALL_FIELDNAMES).writeheader()


def append_ball_label(
    path: Path,
    *,
    bag: str,
    liczba_pilek: str,
    polka: str,
    swiatlo: str,
    czas_s: str,
    uwagi: str,
) -> None:
    """Append one ball recording row to ``path`` (file must already exist)."""
    row = {
        'bag': bag,
        'liczba_pilek': liczba_pilek,
        'polka': polka,
        'swiatlo': swiatlo,
        'czas_s': czas_s,
        'uwagi': uwagi,
    }
    with path.open('a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=BALL_FIELDNAMES).writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_ensure = sub.add_parser(
        'ensure',
        help='Create labels.csv or migrate missing yaw_deg column',
    )
    p_ensure.add_argument('path', type=Path, help='Path to labels.csv')

    p_append = sub.add_parser('append', help='Append one label row')
    p_append.add_argument('path', type=Path, help='Path to labels.csv')
    p_append.add_argument('--bag', required=True)
    p_append.add_argument('--ulozenie', required=True)
    p_append.add_argument('--swiatlo', required=True)
    p_append.add_argument('--czas-s', required=True, dest='czas_s')
    p_append.add_argument('--uwagi', default='')
    p_append.add_argument('--yaw-deg', default='', dest='yaw_deg')

    p_ensure_balls = sub.add_parser(
        'ensure-balls',
        help='Create labels_pilki.csv with header if missing',
    )
    p_ensure_balls.add_argument('path', type=Path, help='Path to labels_pilki.csv')

    p_append_balls = sub.add_parser('append-balls', help='Append one ball label row')
    p_append_balls.add_argument('path', type=Path, help='Path to labels_pilki.csv')
    p_append_balls.add_argument('--bag', required=True)
    p_append_balls.add_argument('--liczba-pilek', required=True, dest='liczba_pilek')
    p_append_balls.add_argument('--polka', required=True)
    p_append_balls.add_argument('--swiatlo', required=True)
    p_append_balls.add_argument('--czas-s', required=True, dest='czas_s')
    p_append_balls.add_argument('--uwagi', default='')

    args = parser.parse_args(argv)
    if args.cmd == 'ensure':
        ensure_labels_csv(args.path)
    elif args.cmd == 'append':
        append_label(
            args.path,
            bag=args.bag,
            ulozenie=args.ulozenie,
            swiatlo=args.swiatlo,
            czas_s=args.czas_s,
            uwagi=args.uwagi,
            yaw_deg=args.yaw_deg,
        )
    elif args.cmd == 'ensure-balls':
        ensure_ball_labels_csv(args.path)
    elif args.cmd == 'append-balls':
        append_ball_label(
            args.path,
            bag=args.bag,
            liczba_pilek=args.liczba_pilek,
            polka=args.polka,
            swiatlo=args.swiatlo,
            czas_s=args.czas_s,
            uwagi=args.uwagi,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
