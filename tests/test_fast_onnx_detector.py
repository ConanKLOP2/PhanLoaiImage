from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fast_onnx_detector import FastOnnxNudeDetector


def write_test_image(path, image):
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    buffer.tofile(str(path))


def test_cv2_unicode_read_path(tmp_path):
    path = tmp_path / "ảnh unicode.jpg"
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    write_test_image(path, image)

    data = np.fromfile(str(path), dtype=np.uint8)
    decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    assert decoded is not None
    assert decoded.shape[:2] == (32, 48)


def test_fast_onnx_preprocess_reads_unicode_filename(tmp_path):
    path = tmp_path / "ảnh unicode.jpg"
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    write_test_image(path, image)

    detector = FastOnnxNudeDetector(
        providers=["CPUExecutionProvider"],
        preprocess_workers=1,
    )
    try:
        blob, metadata = detector._read_and_preprocess(path)
    finally:
        detector.close()

    assert blob.shape == (1, 3, 320, 320)
    assert metadata[2:] == (48, 32)
