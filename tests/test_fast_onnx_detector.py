from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fast_onnx_detector import (
    FastOnnxNudeDetector,
    jpeg_dimensions,
    reduction_factor,
)
from reference_impl import reference_postprocess


def encode(image, extension=".jpg"):
    ok, buffer = cv2.imencode(extension, image)
    assert ok
    return buffer


def write_test_image(path, image):
    encode(image).tofile(str(path))


def noisy_image(width, height, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def detector():
    instance = FastOnnxNudeDetector(
        providers=["CPUExecutionProvider"], preprocess_workers=2
    )
    yield instance
    instance.close()


def test_cv2_unicode_read_path(tmp_path):
    path = tmp_path / "ảnh unicode.jpg"
    write_test_image(path, np.zeros((32, 48, 3), dtype=np.uint8))

    data = np.fromfile(str(path), dtype=np.uint8)
    decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    assert decoded is not None
    assert decoded.shape[:2] == (32, 48)


def test_fast_onnx_preprocess_reads_unicode_filename(tmp_path, detector):
    path = tmp_path / "ảnh unicode.jpg"
    write_test_image(path, np.zeros((32, 48, 3), dtype=np.uint8))

    blob, metadata = detector._read_and_preprocess(path)

    assert blob.shape == (1, 3, 320, 320)
    assert metadata[2:] == (48, 32)


# --- vectorised post-processing --------------------------------------------


def random_prediction(rng, anchors=2100, classes=18, hot_fraction=0.02):
    """Fake raw model output (4 + classes, anchors) with a few confident anchors."""
    prediction = rng.uniform(0.0, 0.15, size=(4 + classes, anchors)).astype(np.float32)
    prediction[0:2] = rng.uniform(0, 320, size=(2, anchors))
    prediction[2:4] = rng.uniform(4, 160, size=(2, anchors))
    hot = rng.random(anchors) < hot_fraction
    prediction[4:, hot] = rng.uniform(0.0, 0.95, size=(classes, int(hot.sum())))
    return prediction


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    "size", [(640, 480), (480, 640), (320, 320), (4000, 3000), (50, 70)]
)
def test_vectorised_postprocess_matches_reference(detector, seed, size):
    width, height = size
    longest = max(width, height)
    metadata = (longest - width, longest - height, width, height)
    prediction = random_prediction(np.random.default_rng(seed))

    expected = reference_postprocess([prediction[None]], metadata)
    actual = detector._postprocess(prediction, metadata)

    assert [d["class"] for d in actual] == [d["class"] for d in expected]
    assert [d["box"] for d in actual] == [d["box"] for d in expected]
    assert [d["score"] for d in actual] == pytest.approx(
        [d["score"] for d in expected], abs=1e-6
    )


def test_postprocess_returns_empty_when_nothing_confident(detector):
    prediction = np.full((22, 2100), 0.05, dtype=np.float32)

    assert detector._postprocess(prediction, (0, 0, 100, 100)) == []


def test_postprocess_clamps_boxes_to_image(detector):
    prediction = np.zeros((22, 1), dtype=np.float32)
    prediction[:4, 0] = (310, 310, 200, 200)  # spills over every edge
    prediction[4 + 2, 0] = 0.9  # BUTTOCKS_EXPOSED

    (detection,) = detector._postprocess(prediction, (0, 0, 320, 320))

    x, y, w, h = detection["box"]
    assert x >= 0 and y >= 0
    assert x + w <= 320 and y + h <= 320
    assert detection["class"] == "BUTTOCKS_EXPOSED"


# --- real model: batching and error isolation ------------------------------


def test_batch_results_match_single_results(tmp_path, detector):
    paths = []
    for index in range(4):
        path = tmp_path / f"img{index}.jpg"
        write_test_image(path, noisy_image(200 + 30 * index, 150, seed=index))
        paths.append(path)

    batched = detector.detect_batch(paths)
    singles = [detector.detect(path) for path in paths]

    assert len(batched) == 4
    for left, right in zip(batched, singles):
        assert [d["class"] for d in left] == [d["class"] for d in right]
        assert [d["score"] for d in left] == pytest.approx(
            [d["score"] for d in right], abs=1e-4
        )


def test_one_bad_image_does_not_fail_the_batch(tmp_path, detector):
    good_a = tmp_path / "a.jpg"
    good_b = tmp_path / "b.jpg"
    corrupt = tmp_path / "corrupt.jpg"
    missing = tmp_path / "missing.jpg"
    write_test_image(good_a, noisy_image(120, 100, seed=1))
    write_test_image(good_b, noisy_image(90, 140, seed=2))
    corrupt.write_bytes(b"this is not an image")

    results = detector.run_prepared(
        detector.prepare_batch([good_a, corrupt, missing, good_b])
    )

    assert isinstance(results[0], list)
    assert isinstance(results[1], ValueError)
    assert isinstance(results[2], FileNotFoundError)
    assert isinstance(results[3], list)
    expected_a = detector.detect(good_a)
    assert [d["class"] for d in results[0]] == [d["class"] for d in expected_a]


def test_batch_of_only_bad_images_skips_inference(tmp_path, detector):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"nope")

    results = detector.run_prepared(detector.prepare_batch([corrupt]))

    assert len(results) == 1 and isinstance(results[0], ValueError)


def test_detect_batch_still_raises_for_bad_image(tmp_path, detector):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"nope")

    with pytest.raises(ValueError):
        detector.detect_batch([corrupt])


def test_empty_batch(detector):
    assert detector.detect_batch([]) == []


def test_preprocess_blob_matches_blob_from_image(tmp_path, detector):
    path = tmp_path / "x.png"
    image = noisy_image(200, 120, seed=3)
    encode(image, ".png").tofile(str(path))

    blob, metadata = detector._read_and_preprocess(path)

    # Same colour conversion the original (NudeNet-compatible) code applied.
    converted = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    padded = cv2.copyMakeBorder(converted, 0, 80, 0, 0, cv2.BORDER_CONSTANT)
    expected = cv2.dnn.blobFromImage(
        padded, 1 / 255.0, (320, 320), (0, 0, 0), swapRB=True, crop=False
    )
    np.testing.assert_array_equal(blob, expected)
    assert metadata == (0, 80, 200, 120)


# --- JPEG header parsing and reduced decode --------------------------------


@pytest.mark.parametrize("size", [(640, 480), (33, 77), (1600, 1200), (1, 1)])
def test_jpeg_dimensions_reads_header(size):
    width, height = size
    data = encode(np.zeros((height, width, 3), dtype=np.uint8))

    assert jpeg_dimensions(data) == (width, height)


def test_jpeg_dimensions_rejects_non_jpeg():
    assert jpeg_dimensions(b"") is None
    assert jpeg_dimensions(b"\x89PNG\r\n\x1a\n" + bytes(40)) is None
    assert jpeg_dimensions(encode(np.zeros((8, 8, 3), np.uint8), ".png")) is None
    assert jpeg_dimensions(b"\xff\xd8\xff") is None  # truncated


@pytest.mark.parametrize(
    "width, height, expected",
    [
        (320, 320, 1),
        (639, 100, 1),
        (640, 480, 2),
        (1280, 720, 4),
        (2560, 1440, 8),
        (6000, 4000, 8),
        (5000, 200, 8),
    ],
)
def test_reduction_factor_keeps_long_side_at_least_target(width, height, expected):
    factor = reduction_factor(width, height, 320)

    assert factor == expected
    assert max(width, height) // factor >= 320 or factor == 1


def test_fast_decode_reduces_large_jpeg_and_keeps_original_metadata(tmp_path):
    path = tmp_path / "big.jpg"
    # Smooth content so reduced decode and full decode agree closely.
    gradient = np.tile(np.linspace(0, 255, 1600, dtype=np.uint8), (1200, 1))
    write_test_image(path, np.stack([gradient, gradient[::-1], gradient], axis=2))

    full = FastOnnxNudeDetector(providers=["CPUExecutionProvider"], preprocess_workers=1)
    fast = FastOnnxNudeDetector(
        providers=["CPUExecutionProvider"], preprocess_workers=1, fast_decode=True
    )
    try:
        full_blob, full_meta = full._read_and_preprocess(path)
        fast_blob, fast_meta = fast._read_and_preprocess(path)
    finally:
        full.close()
        fast.close()

    assert fast_blob.shape == full_blob.shape == (1, 3, 320, 320)
    assert fast_meta == full_meta == (0, 400, 1600, 1200)
    assert np.abs(fast_blob - full_blob).mean() < 0.02


def test_fast_decode_falls_back_for_small_and_non_jpeg(tmp_path):
    small = tmp_path / "small.jpg"
    png = tmp_path / "big.png"
    write_test_image(small, noisy_image(200, 150))
    encode(noisy_image(1600, 1200), ".png").tofile(str(png))

    full = FastOnnxNudeDetector(providers=["CPUExecutionProvider"], preprocess_workers=1)
    fast = FastOnnxNudeDetector(
        providers=["CPUExecutionProvider"], preprocess_workers=1, fast_decode=True
    )
    try:
        for path in (small, png):
            np.testing.assert_array_equal(
                fast._read_and_preprocess(path)[0], full._read_and_preprocess(path)[0]
            )
    finally:
        full.close()
        fast.close()
