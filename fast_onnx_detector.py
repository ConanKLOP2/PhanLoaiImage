from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

SCORE_FLOOR = 0.2
NMS_SCORE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45

_JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_JPEG_STANDALONE_MARKERS = frozenset({0x01, 0xD8} | set(range(0xD0, 0xD8)))


def jpeg_dimensions(data: np.ndarray | bytes) -> tuple[int, int] | None:
    """Return (width, height) from a JPEG header without decoding, or None."""
    buffer = memoryview(data)
    size = len(buffer)
    if size < 4 or buffer[0] != 0xFF or buffer[1] != 0xD8:
        return None

    index = 2
    while index + 4 <= size:
        if buffer[index] != 0xFF:
            return None
        marker = buffer[index + 1]
        if marker == 0xFF:
            index += 1
            continue
        if marker in _JPEG_STANDALONE_MARKERS:
            index += 2
            continue
        if marker == 0xDA:
            return None
        if marker in _JPEG_SOF_MARKERS:
            if index + 9 > size:
                return None
            height = (buffer[index + 5] << 8) | buffer[index + 6]
            width = (buffer[index + 7] << 8) | buffer[index + 8]
            if width == 0 or height == 0:
                return None
            return width, height
        index += 2 + ((buffer[index + 2] << 8) | buffer[index + 3])
    return None


def reduction_factor(width: int, height: int, target: int) -> int:
    """Largest JPEG DCT downscale (1, 2, 4 or 8) that keeps the long side >= target."""
    long_side = max(width, height)
    for factor in (8, 4, 2):
        if long_side // factor >= target and min(width, height) // factor >= 1:
            return factor
    return 1


@dataclass
class PreparedBatch:
    """Images submitted for preprocessing; resolve them with run_prepared()."""

    paths: list
    futures: list[Future]


class FastOnnxNudeDetector:
    accepts_unicode_paths = True

    def __init__(
        self,
        model_path: Path | None = None,
        providers: list[str] | None = None,
        inference_resolution: int = 320,
        preprocess_workers: int = 4,
        fast_decode: bool = False,
    ) -> None:
        import onnxruntime as ort

        if model_path is None:
            import nudenet

            model_path = Path(nudenet.__file__).resolve().parent / "320n.onnx"

        if providers and "CUDAExecutionProvider" in providers:
            try:
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
            except Exception:
                pass
            try:
                import torch  # noqa: F401
            except Exception:
                pass

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            os.fspath(model_path),
            sess_options=session_options,
            providers=providers,
        )
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.input_width = inference_resolution
        self.input_height = inference_resolution
        self.fast_decode = fast_decode
        self.preprocess_workers = max(1, preprocess_workers)
        self.executor = ThreadPoolExecutor(max_workers=self.preprocess_workers)

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def detect(self, image_path: str | Path) -> list[dict]:
        return self.detect_batch([image_path])[0]

    def detect_batch(
        self,
        image_paths: Iterable[str | Path],
        batch_size: int | None = None,
    ) -> list[list[dict]]:
        """Detect a batch; raises the first per-image error, like the old API."""
        results = self.run_prepared(self.prepare_batch(image_paths))
        for result in results:
            if isinstance(result, Exception):
                raise result
        return results  # type: ignore[return-value]

    def prepare_batch(self, image_paths: Iterable[str | Path]) -> PreparedBatch:
        """Start reading/decoding images on the worker pool and return at once."""
        paths = list(image_paths)
        futures = [self.executor.submit(self._read_and_preprocess, path) for path in paths]
        return PreparedBatch(paths=paths, futures=futures)

    def run_prepared(self, prepared: PreparedBatch) -> list[list[dict] | Exception]:
        """Run inference on a prepared batch.

        An image that cannot be read or decoded yields its exception in its own
        slot; the other images in the batch are still inferred together.
        """
        results: list[list[dict] | Exception] = [[] for _ in prepared.paths]
        good: list[tuple[int, tuple]] = []
        for index, future in enumerate(prepared.futures):
            try:
                good.append((index, future.result()))
            except Exception as exc:
                results[index] = exc

        if not good:
            return results

        batch_input = np.empty(
            (len(good), 3, self.input_height, self.input_width), dtype=np.float32
        )
        for slot, (_, (blob, _)) in enumerate(good):
            batch_input[slot] = blob[0]

        outputs = self.session.run(None, {self.input_name: batch_input})[0]
        for slot, (index, (_, metadata)) in enumerate(good):
            results[index] = self._postprocess(outputs[slot], metadata)
        return results

    def _read_and_preprocess(self, image_path: str | Path) -> tuple[np.ndarray, tuple]:
        data = np.fromfile(os.fspath(image_path), dtype=np.uint8)

        if self.fast_decode:
            fast = self._fast_decode_jpeg(data)
            if fast is not None:
                return self._to_blob(*fast)

        mat = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if mat is None:
            raise ValueError(f"Khong doc duoc anh: {image_path}")

        image_original_width = mat.shape[1]
        image_original_height = mat.shape[0]

        if len(mat.shape) == 2:
            mat_c3 = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
        else:
            # Match NudeNet's preprocessing exactly for color images.
            mat_c3 = cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)

        return self._to_blob(mat_c3, image_original_width, image_original_height)

    def _fast_decode_jpeg(self, data: np.ndarray) -> tuple[np.ndarray, int, int] | None:
        dimensions = jpeg_dimensions(data)
        if dimensions is None:
            return None
        width, height = dimensions
        factor = reduction_factor(width, height, max(self.input_width, self.input_height))
        if factor == 1:
            return None
        flags = {
            2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4,
            8: cv2.IMREAD_REDUCED_COLOR_8,
        }[factor] | cv2.IMREAD_IGNORE_ORIENTATION
        mat = cv2.imdecode(data, flags)
        if mat is None:
            return None
        return mat, width, height

    def _to_blob(
        self, mat_c3: np.ndarray, original_width: int, original_height: int
    ) -> tuple[np.ndarray, tuple]:
        """Pad to a square and build the NCHW blob.

        mat_c3 may be smaller than the original (reduced decode); padding is
        derived from it while the metadata keeps original-image coordinates.
        """
        max_size = max(mat_c3.shape[:2])
        mat_pad = cv2.copyMakeBorder(
            mat_c3,
            0,
            max_size - mat_c3.shape[0],
            0,
            max_size - mat_c3.shape[1],
            cv2.BORDER_CONSTANT,
        )
        input_blob = cv2.dnn.blobFromImage(
            mat_pad,
            1 / 255.0,
            (self.input_width, self.input_height),
            (0, 0, 0),
            swapRB=True,
            crop=False,
        )

        original_max = max(original_width, original_height)
        metadata = (
            original_max - original_width,
            original_max - original_height,
            original_width,
            original_height,
        )
        return input_blob, metadata

    def _postprocess(self, prediction: np.ndarray, metadata: tuple) -> list[dict]:
        """Turn one image's raw model output (4 + classes, anchors) into detections."""
        x_pad, y_pad, image_width, image_height = metadata
        rows = prediction.T
        class_scores = rows[:, 4:]
        max_scores = class_scores.max(axis=1)

        keep = max_scores >= SCORE_FLOOR
        if not keep.any():
            return []

        rows = rows[keep]
        max_scores = max_scores[keep]
        class_ids = class_scores[keep].argmax(axis=1)

        cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        x = cx - w / 2
        y = cy - h / 2

        x = x * (image_width + x_pad) / self.input_width
        y = y * (image_height + y_pad) / self.input_height
        w = w * (image_width + x_pad) / self.input_width
        h = h * (image_height + y_pad) / self.input_height

        x = np.clip(x, 0, image_width)
        y = np.clip(y, 0, image_height)
        w = np.minimum(w, image_width - x)
        h = np.minimum(h, image_height - y)

        boxes = np.stack([x, y, w, h], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            max_scores.tolist(),
            NMS_SCORE_THRESHOLD,
            NMS_IOU_THRESHOLD,
        )
        if len(indices) == 0:
            return []

        detections = []
        for index in np.array(indices).flatten():
            bx, by, bw, bh = boxes[index]
            detections.append(
                {
                    "class": LABELS[int(class_ids[index])],
                    "score": float(max_scores[index]),
                    "box": [int(bx), int(by), int(bw), int(bh)],
                }
            )
        return detections
