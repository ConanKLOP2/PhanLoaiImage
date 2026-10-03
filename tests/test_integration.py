"""End-to-end runs through the real ONNX model on CPU."""
from __future__ import annotations

import csv

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")
pytest.importorskip("nudenet")

import bench
import classify_images


def make_images(folder, count=6):
    folder.mkdir(parents=True, exist_ok=True)
    paths = bench.make_synthetic(folder, count)
    (folder / "corrupt.jpg").write_bytes(b"definitely not a jpeg")
    return paths


def test_real_model_end_to_end_isolates_corrupt_image(tmp_path):
    make_images(tmp_path / "set")

    (_, result), = classify_images.scan_folders(
        [tmp_path / "set"],
        device="cpu",
        mode="copy",
        batch_size=4,
        preprocess_workers=2,
        transfer_workers=1,
    )

    assert result.total_seen == 7
    assert result.processed == 6
    assert result.errors == 1
    assert result.batch_errors == 0  # the corrupt file did not poison its batch
    assert (tmp_path / "set" / "_classified" / "errors" / "corrupt.jpg").exists()
    with (tmp_path / "set" / "_classified" / "manifest.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 7


def test_real_model_run_is_resumable_and_deterministic(tmp_path):
    make_images(tmp_path / "set", count=4)
    options = dict(device="cpu", mode="copy", batch_size=2, preprocess_workers=2, transfer_workers=1)

    first = classify_images.scan_folders([tmp_path / "set"], **options)[0][1]
    second = classify_images.scan_folders([tmp_path / "set"], **options)[0][1]

    assert first.processed == 4 and first.errors == 1
    # Errors are retried on a rerun; good images are not reprocessed.
    assert second.processed == 0


def test_fast_decode_gives_same_categories_as_full_decode(tmp_path):
    paths = bench.make_synthetic(tmp_path, 8)
    from fast_onnx_detector import FastOnnxNudeDetector
    from policy import classify_detection

    def categories(fast):
        detector = FastOnnxNudeDetector(
            providers=["CPUExecutionProvider"], preprocess_workers=2, fast_decode=fast
        )
        try:
            return [
                classify_detection(result, 0.55, 0.55)[0]
                for result in detector.detect_batch(paths)
            ]
        finally:
            detector.close()

    assert categories(True) == categories(False)


def test_bench_golden_round_trip(tmp_path):
    import json

    golden = tmp_path / "golden.json"
    set_dir = tmp_path / "set"
    bench.make_synthetic(set_dir, 6)
    args = ["--dir", str(set_dir), "--batch-size", "3", "--device", "cpu"]

    assert bench.main(args + ["--save-golden", str(golden)]) == 0
    assert bench.main(args + ["--compare-golden", str(golden)]) == 0

    saved = json.loads(golden.read_text(encoding="utf-8"))
    assert len(saved) == 6
    assert set(saved.values()) <= {"nude", "sexy", "normal"}


def test_bench_golden_detects_changed_categories(tmp_path):
    import json

    set_dir = tmp_path / "set"
    paths = bench.make_synthetic(set_dir, 4)
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps({str(p): "nude" for p in paths}), encoding="utf-8"
    )

    code = bench.main(
        ["--dir", str(set_dir), "--batch-size", "2", "--device", "cpu",
         "--compare-golden", str(golden)]
    )

    assert code == 2
