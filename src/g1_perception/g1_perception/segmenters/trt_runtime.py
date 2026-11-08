"""
Run and build TensorRT YOLO engines without PyTorch (TensorRT + cudart).

On the robot (JetPack 5, Python 3.10) there is no CUDA-enabled torch, so
Ultralytics cannot open ``.engine`` files. :class:`TrtYoloModel` runs the
engine through the ``tensorrt`` bindings, moves buffers with ``libcudart``
(ctypes) and decodes outputs with :mod:`yolo_decode`. Engine files use the
Ultralytics layout (4-byte little-endian metadata length, JSON metadata,
serialized engine), so engines built here or by ``yolo export`` both load.
"""

from __future__ import annotations

import ast
import ctypes
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from g1_perception.segmenters.yolo_decode import YoloResult, decode, letterbox

_CUDART_NAMES = ('libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12')
_H2D, _D2H = 1, 2  # cudaMemcpyKind


@dataclass(frozen=True)
class EngineMeta:
    """Model metadata stored in front of the engine (Ultralytics format)."""

    names: dict[int, str]
    task: str
    imgsz: tuple[int, int]
    kpt_shape: tuple[int, int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineMeta:
        """Parse Ultralytics metadata (JSON keys are strings)."""
        names = data.get('names') or {}
        if isinstance(names, str):
            names = ast.literal_eval(names)
        if isinstance(names, list):
            names = dict(enumerate(names))
        imgsz = data.get('imgsz', [640, 640])
        if isinstance(imgsz, str):
            imgsz = ast.literal_eval(imgsz)
        if isinstance(imgsz, int):
            imgsz = [imgsz, imgsz]
        kpt = data.get('kpt_shape')
        if isinstance(kpt, str):
            kpt = ast.literal_eval(kpt)
        return cls(
            names={int(k): str(v) for k, v in names.items()},
            task=str(data.get('task', 'detect')),
            imgsz=(int(imgsz[0]), int(imgsz[1])),
            kpt_shape=None if kpt is None else (int(kpt[0]), int(kpt[1])),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable metadata (Ultralytics keys)."""
        out: dict[str, Any] = {
            'names': {str(k): v for k, v in self.names.items()},
            'task': self.task,
            'imgsz': list(self.imgsz),
            'batch': 1,
        }
        if self.kpt_shape is not None:
            out['kpt_shape'] = list(self.kpt_shape)
        return out


def split_engine_file(data: bytes) -> tuple[dict[str, Any] | None, bytes]:
    """Split ``<len><json><engine>``; return ``(None, data)`` without header."""
    if len(data) > 4:
        meta_len = int.from_bytes(data[:4], byteorder='little')
        if 0 < meta_len < min(len(data) - 4, 1 << 20):
            try:
                meta = json.loads(data[4:4 + meta_len].decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                meta = None
            if isinstance(meta, dict):
                return meta, data[4 + meta_len:]
    return None, data


def join_engine_file(meta: EngineMeta, engine: bytes) -> bytes:
    """Prefix a serialized engine with Ultralytics-style metadata."""
    blob = json.dumps(meta.to_dict()).encode('utf-8')
    return len(blob).to_bytes(4, byteorder='little') + blob + engine


def meta_from_onnx(onnx_path: str | Path) -> EngineMeta:
    """Read Ultralytics metadata (names, task, imgsz, kpt_shape) from ONNX."""
    import onnx

    props = {p.key: p.value for p in onnx.load(str(onnx_path)).metadata_props}
    if 'names' not in props:
        raise ValueError(f'{onnx_path} has no Ultralytics metadata (names)')
    return EngineMeta.from_dict(props)


def cuda_driver_ok() -> bool:
    """Check that ``libcuda`` initialises and reports at least one GPU."""
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
    except OSError:
        return False
    if cuda.cuInit(0) != 0:
        return False
    count = ctypes.c_int(0)
    return cuda.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0


def tensorrt_available() -> bool:
    """Check that ``import tensorrt`` works and a CUDA GPU is usable."""
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        return False
    return cuda_driver_ok()


class _CudaRuntime:
    """Minimal ``libcudart`` wrapper: device buffers, one stream, copies."""

    def __init__(self) -> None:
        last: OSError | None = None
        for name in _CUDART_NAMES:
            try:
                self._lib = ctypes.CDLL(name)
                break
            except OSError as exc:
                last = exc
        else:
            raise RuntimeError(f'libcudart not found ({last})')
        self._lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.stream = ctypes.c_void_p()
        self._check(self._lib.cudaStreamCreate(ctypes.byref(self.stream)))

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self._check(self._lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
        return int(ptr.value)

    def free(self, ptr: int) -> None:
        self._lib.cudaFree(ctypes.c_void_p(ptr))

    def to_device(self, dst: int, src: NDArray) -> None:
        self._check(self._lib.cudaMemcpyAsync(
            ctypes.c_void_p(dst), src.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(src.nbytes), _H2D, self.stream,
        ))

    def to_host(self, dst: NDArray, src: int) -> None:
        self._check(self._lib.cudaMemcpyAsync(
            dst.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src),
            ctypes.c_size_t(dst.nbytes), _D2H, self.stream,
        ))

    def synchronize(self) -> None:
        self._check(self._lib.cudaStreamSynchronize(self.stream))

    def _check(self, code: int) -> None:
        if code != 0:
            msg = self._lib.cudaGetErrorString(code).decode()
            raise RuntimeError(f'CUDA error {code}: {msg}')


class TrtEngineRunner:
    """
    Deserialize an engine and run it on one input array.

    Outputs are returned in binding-name order (``output0``, ``output1``),
    which matches the Ultralytics ONNX export.
    """

    def __init__(self, engine_bytes: bytes) -> None:
        import tensorrt as trt

        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        self._engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError('TensorRT could not deserialize the engine')
        self._context = self._engine.create_execution_context()
        self._cuda = _CudaRuntime()
        self._ptrs: list[int] = []
        self._inputs: list[tuple[int, NDArray]] = []
        self._outputs: list[tuple[str, int, NDArray]] = []
        for i in range(self._engine.num_bindings):
            shape = tuple(self._engine.get_binding_shape(i))
            dtype = np.dtype(trt.nptype(self._engine.get_binding_dtype(i)))
            host = np.empty(shape, dtype=dtype)
            ptr = self._cuda.malloc(host.nbytes)
            self._ptrs.append(ptr)
            if self._engine.binding_is_input(i):
                self._inputs.append((ptr, host))
            else:
                self._outputs.append((self._engine.get_binding_name(i), ptr, host))
        self._outputs.sort(key=lambda o: o[0])
        if len(self._inputs) != 1:
            raise RuntimeError(f'Expected 1 input binding, got {len(self._inputs)}')

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self._inputs[0][1].shape)

    def infer(self, blob: NDArray) -> list[NDArray]:
        """Copy ``blob`` in, execute, return copies of all outputs."""
        ptr, host = self._inputs[0]
        src = np.ascontiguousarray(blob, dtype=host.dtype)
        if src.shape != host.shape:
            raise ValueError(f'Input shape {src.shape} != engine {host.shape}')
        self._cuda.to_device(ptr, src)
        if not self._context.execute_async_v2(self._ptrs, self._cuda.stream.value):
            raise RuntimeError('TensorRT execution failed')
        for _name, out_ptr, out_host in self._outputs:
            self._cuda.to_host(out_host, out_ptr)
        self._cuda.synchronize()
        return [out_host.astype(np.float32, copy=True) for _n, _p, out_host in self._outputs]

    def __del__(self) -> None:
        for ptr in getattr(self, '_ptrs', []):
            self._cuda.free(ptr)


class TrtYoloModel:
    """
    Ultralytics-like ``predict`` over a TensorRT engine (no torch).

    ``predict(bgr, conf=, iou=, max_det=)`` returns ``[YoloResult]``;
    other Ultralytics keywords (``device``, ``retina_masks``, ``verbose``,
    ``imgsz``) are accepted and ignored — masks are always full size.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        runner_factory: Callable[[bytes], Any] = TrtEngineRunner,
    ) -> None:
        data = Path(path).read_bytes()
        meta, engine = split_engine_file(data)
        if meta is None:
            raise RuntimeError(
                f'{path}: no metadata header (build it with tools/export_model.py '
                'or `yolo export format=engine`)'
            )
        self.meta = EngineMeta.from_dict(meta)
        self.names = dict(self.meta.names)
        self.task = self.meta.task
        self._runner = runner_factory(engine)

    def predict(
        self,
        source: NDArray[np.uint8],
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        **_ignored: Any,
    ) -> list[YoloResult]:
        """Run one BGR image and decode it like Ultralytics."""
        blob, info = letterbox(source, self.meta.imgsz)
        outputs = self._runner.infer(blob)
        return [decode(
            outputs, info, task=self.task, names=self.names,
            kpt_shape=self.meta.kpt_shape, conf=conf, iou=iou, max_det=max_det,
        )]


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    half: bool = True,
    workspace_gb: float = 1.0,
    meta: EngineMeta | None = None,
) -> Path:
    """
    Build a TensorRT engine from ONNX with the TensorRT API (no torch).

    FP16 when ``half`` and the GPU supports it. The metadata (from the ONNX
    file unless given) is written in front of the engine.
    """
    import tensorrt as trt

    meta = meta or meta_from_onnx(onnx_path)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f'ONNX parse failed: {errors}')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )
    if half and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')
    out = Path(engine_path)
    out.write_bytes(join_engine_file(meta, bytes(serialized)))
    return out
