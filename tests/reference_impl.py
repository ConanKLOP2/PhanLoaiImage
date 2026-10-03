"""Pre-optimisation reference implementations, kept to prove equivalence."""
from __future__ import annotations

import cv2
import numpy as np

from fast_onnx_detector import LABELS


def reference_postprocess(output, metadata, input_width=320, input_height=320):
    """The original per-row Python loop from FastOnnxNudeDetector._postprocess."""
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

            x = x * (image_original_width + x_pad) / input_width
            y = y * (image_original_height + y_pad) / input_height
            w = w * (image_original_width + x_pad) / input_width
            h = h * (image_original_height + y_pad) / input_height

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


def reference_preprocess(path, input_width=320, input_height=320):
    """The original preprocessing: RGBA2BGR conversion, pad, blobFromImage(swapRB)."""
    import os

    data = np.fromfile(os.fspath(path), dtype=np.uint8)
    mat = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    width, height = mat.shape[1], mat.shape[0]
    if len(mat.shape) == 2:
        mat_c3 = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
    else:
        mat_c3 = cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)
    max_size = max(mat_c3.shape[:2])
    x_pad = max_size - mat_c3.shape[1]
    y_pad = max_size - mat_c3.shape[0]
    padded = cv2.copyMakeBorder(mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)
    blob = cv2.dnn.blobFromImage(
        padded, 1 / 255.0, (input_width, input_height), (0, 0, 0), swapRB=True, crop=False
    )
    return blob, (x_pad, y_pad, width, height)
