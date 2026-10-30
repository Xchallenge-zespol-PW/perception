#!/usr/bin/env python3
"""
Live YOLO-seg demo from a webcam (OpenCV).

Pip OpenCV here is headless (no ``imshow``), so the default view is an
MJPEG stream in the browser. Overlay matches Ultralytics predict
(masks + class name + confidence).

Example::

    python3 tools/webcam_yolo.py
    # open http://127.0.0.1:8765/

    python3 tools/webcam_yolo.py --camera 0 --conf 0.4
    python3 tools/webcam_yolo.py --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DEFAULT_WEIGHTS = Path('runs/segment/runs/seg/cubes-2/weights/best.pt')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Open a webcam with OpenCV and run YOLO-seg live. Default view '
            'is a browser MJPEG stream (OpenCV GUI is unavailable here).'
        )
    )
    parser.add_argument(
        '--weights',
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f'Model .pt / .engine (default: {DEFAULT_WEIGHTS})',
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='OpenCV camera index (default: 0)',
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.4,
        help='Confidence threshold (default: 0.4)',
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='NMS IoU threshold (default: 0.7)',
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Inference size (default: 640)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Ultralytics device, e.g. cpu / 0 (default: auto)',
    )
    parser.add_argument(
        '--width',
        type=int,
        default=0,
        help='Force capture width (0 = leave default)',
    )
    parser.add_argument(
        '--height',
        type=int,
        default=0,
        help='Force capture height (0 = leave default)',
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='MJPEG HTTP bind host (default: 127.0.0.1)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='MJPEG HTTP port (default: 8765)',
    )
    return parser.parse_args(argv)


class _FrameHub:
    """Thread-safe latest JPEG for the HTTP stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._conf = 0.4
        self._stop = False

    @property
    def stop(self) -> bool:
        return self._stop

    def request_stop(self) -> None:
        self._stop = True

    def set_jpeg(self, data: bytes) -> None:
        with self._lock:
            self._jpeg = data

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def set_conf(self, conf: float) -> None:
        with self._lock:
            self._conf = float(conf)

    def get_conf(self) -> float:
        with self._lock:
            return self._conf


def _make_handler(hub: _FrameHub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split('?', 1)[0]
            if path in ('/', '/index.html'):
                body = (
                    '<!doctype html><html><head><meta charset="utf-8">'
                    '<title>webcam_yolo</title>'
                    '<style>body{margin:0;background:#111;color:#ddd;'
                    'font:14px sans-serif}'
                    'img{display:block;max-width:100%;height:auto;margin:auto}'
                    '.bar{padding:8px 12px}</style></head><body>'
                    '<div class="bar">YOLO-seg live - '
                    '<a href="/conf?d=-0.05" style="color:#8f8">conf -</a> | '
                    '<a href="/conf?d=0.05" style="color:#8f8">conf +</a> | '
                    '<a href="/quit" style="color:#f88">quit</a></div>'
                    '<img src="/stream.mjpg" alt="stream">'
                    '</body></html>'
                ).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == '/conf':
                delta = 0.0
                if '?' in self.path:
                    for part in self.path.split('?', 1)[1].split('&'):
                        if part.startswith('d='):
                            try:
                                delta = float(part[2:])
                            except ValueError:
                                delta = 0.0
                conf = min(0.95, max(0.05, hub.get_conf() + delta))
                hub.set_conf(conf)
                msg = f'conf={conf:.2f}\n'.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return

            if path == '/quit':
                hub.request_stop()
                msg = b'stopping\n'
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return

            if path != '/stream.mjpg':
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header(
                'Content-Type',
                'multipart/x-mixed-replace; boundary=frame',
            )
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            try:
                while not hub.stop:
                    jpeg = hub.get_jpeg()
                    if jpeg is None:
                        time.sleep(0.02)
                        continue
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(
                        f'Content-Length: {len(jpeg)}\r\n\r\n'.encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.weights.is_file():
        print(f'ERROR: weights not found: {args.weights}', file=sys.stderr)
        return 1

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        print(f'ERROR: missing dependency: {exc}', file=sys.stderr)
        return 1

    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'ERROR: cannot open camera index {args.camera}', file=sys.stderr)
        return 1

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    predict_kwargs: dict = {
        'retina_masks': True,
        'conf': args.conf,
        'iou': args.iou,
        'imgsz': args.imgsz,
        'verbose': False,
    }
    if args.device is not None:
        predict_kwargs['device'] = args.device

    hub = _FrameHub()
    hub.set_conf(float(args.conf))

    server = ThreadingHTTPServer((args.host, args.port), _make_handler(hub))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f'http://{args.host}:{args.port}/'
    print(
        f'Loaded {args.weights}\n'
        f'Camera {args.camera}  conf={args.conf}  iou={args.iou}\n'
        f'Open browser: {url}\n'
        'Ctrl+C to quit (or /quit in the page).'
    )

    fps_ema = 0.0
    try:
        while not hub.stop:
            ok, frame = cap.read()
            if not ok:
                print('ERROR: failed to read frame', file=sys.stderr)
                break

            conf = hub.get_conf()
            t0 = time.perf_counter()
            predict_kwargs['conf'] = conf
            results = model.predict(frame, **predict_kwargs)
            dt = time.perf_counter() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = fps if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * fps)

            plotted = results[0].plot()
            n = 0
            if results[0].boxes is not None:
                n = int(len(results[0].boxes))

            hud = f'FPS {fps_ema:.1f}  conf>={conf:.2f}  dets={n}'
            cv2.putText(
                plotted,
                hud,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 220, 20),
                2,
                cv2.LINE_AA,
            )
            ok_j, buf = cv2.imencode(
                '.jpg', plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if ok_j:
                hub.set_jpeg(buf.tobytes())
    except KeyboardInterrupt:
        print('\nStopping.')
    finally:
        hub.request_stop()
        server.shutdown()
        cap.release()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
