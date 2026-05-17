from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
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


class FastOnnxNudeDetector:
    accepts_unicode_paths = True

    def __init__(
        self,
        model_path: Path | None = None,
        providers: list[str] | None = None,
        inference_resolution: int = 320,
        preprocess_workers: int = 4,
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
        self.input_name = self.session.get_inputs()[0].name
        self.input_width = inference_resolution
        self.input_height = inference_resolution
        self.preprocess_workers = max(1, preprocess_workers)

    def detect(self, image_path: str | Path) -> list[dict]:
        return self.detect_batch([image_path])[0]

    def detect_batch(
        self,
        image_paths: Iterable[str | Path],
        batch_size: int | None = None,
    ) -> list[list[dict]]:
        paths = list(image_paths)
        if not paths:
            return []

        with ThreadPoolExecutor(max_workers=self.preprocess_workers) as executor:
            prepared = list(executor.map(self._read_and_preprocess, paths))

        batch_input = np.vstack([item[0] for item in prepared])
        outputs = self.session.run(None, {self.input_name: batch_input})

        results: list[list[dict]] = []
        for index, item in enumerate(prepared):
            _, metadata = item
            results.append(self._postprocess([outputs[0][index : index + 1]], metadata))
        return results

    def _read_and_preprocess(self, image_path: str | Path) -> tuple[np.ndarray, tuple]:
        data = np.fromfile(os.fspath(image_path), dtype=np.uint8)
        mat = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if mat is None:
            raise ValueError(f"Khong doc duoc anh: {image_path}")

        image_original_width = mat.shape[1]
        image_original_height = mat.shape[0]

        if len(mat.shape) == 2:
            mat_c3 = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
        elif mat.shape[2] == 4:
            mat_c3 = cv2.cvtColor(mat, cv2.COLOR_BGRA2BGR)
        else:
            mat_c3 = mat

        max_size = max(mat_c3.shape[:2])
        x_pad = max_size - mat_c3.shape[1]
        y_pad = max_size - mat_c3.shape[0]

        mat_pad = cv2.copyMakeBorder(
            mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT
        )
        input_blob = cv2.dnn.blobFromImage(
            mat_pad,
            1 / 255.0,
            (self.input_width, self.input_height),
            (0, 0, 0),
            swapRB=True,
            crop=False,
        )

        metadata = (
            x_pad,
            y_pad,
            image_original_width,
            image_original_height,
        )
        return input_blob, metadata

    def _postprocess(self, output: list[np.ndarray], metadata: tuple) -> list[dict]:
        x_pad, y_pad, image_original_width, image_original_height = metadata
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            if max_score >= 0.2:
                class_id = int(np.argmax(classes_scores))
                x, y, w, h = outputs[i][0:4]
                x = x - w / 2
                y = y - h / 2

                x = x * (image_original_width + x_pad) / self.input_width
                y = y * (image_original_height + y_pad) / self.input_height
                w = w * (image_original_width + x_pad) / self.input_width
                h = h * (image_original_height + y_pad) / self.input_height

                x = max(0, min(x, image_original_width))
                y = max(0, min(y, image_original_height))
                w = min(w, image_original_width - x)
                h = min(h, image_original_height - y)

                class_ids.append(class_id)
                scores.append(float(max_score))
                boxes.append([float(x), float(y), float(w), float(h)])

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
        if len(indices) == 0:
            return []

        detections = []
        for raw_index in np.array(indices).flatten():
            index = int(raw_index)
            x, y, w, h = boxes[index]
            detections.append(
                {
                    "class": LABELS[class_ids[index]],
                    "score": float(scores[index]),
                    "box": [int(x), int(y), int(w), int(h)],
                }
            )
        return detections
