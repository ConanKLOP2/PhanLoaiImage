from __future__ import annotations

from pathlib import Path

import pytest

import classify_images


class DummyDetector:
    accepts_unicode_paths = True

    def __init__(self) -> None:
        self.closed = False

    def detect_batch(self, paths):
        return [[] for _ in paths]

    def close(self) -> None:
        self.closed = True


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image bytes")


@pytest.fixture()
def dummy_detector(monkeypatch):
    detector = DummyDetector()

    def fake_load_detector(device="auto", engine="onnx", preprocess_workers=4):
        return detector, ["CPUExecutionProvider"]

    monkeypatch.setattr(classify_images, "load_detector", fake_load_detector)
    return detector


def test_transfer_status_values():
    assert classify_images.transfer_status("copy") == "copied"
    assert classify_images.transfer_status("move") == "moved"


def test_root_output_strategy_copies_to_root_classified(tmp_path, dummy_detector):
    write_image(tmp_path / "root.jpg")
    write_image(tmp_path / "sub" / "child.jpg")

    result = classify_images.scan_and_classify(
        tmp_path,
        mode="copy",
        output_strategy="root",
        batch_size=2,
        transfer_workers=1,
    )

    assert result.processed == 2
    assert (tmp_path / "_classified" / "normal" / "root.jpg").exists()
    assert (tmp_path / "_classified" / "normal" / "child.jpg").exists()
    assert not (tmp_path / "sub" / "_classified").exists()
    assert dummy_detector.closed


def test_per_folder_output_strategy_copies_next_to_each_source_folder(
    tmp_path, dummy_detector
):
    write_image(tmp_path / "root.jpg")
    write_image(tmp_path / "sub" / "child.jpg")

    result = classify_images.scan_and_classify(
        tmp_path,
        mode="copy",
        output_strategy="per-folder",
        batch_size=2,
        transfer_workers=1,
    )

    assert result.processed == 2
    assert (tmp_path / "_classified" / "normal" / "root.jpg").exists()
    assert (tmp_path / "sub" / "_classified" / "normal" / "child.jpg").exists()


def test_resume_skips_copied_and_legacy_copyd_status(tmp_path, dummy_detector):
    write_image(tmp_path / "a.jpg")
    manifest_dir = tmp_path / "_classified"
    manifest_dir.mkdir()
    (manifest_dir / "manifest.csv").write_text(
        "source,destination,category,status,reason\n"
        f"{tmp_path / 'a.jpg'},,normal,copyd,no_sensitive_label\n",
        encoding="utf-8",
    )

    result = classify_images.scan_and_classify(
        tmp_path,
        mode="copy",
        output_strategy="root",
        batch_size=1,
        transfer_workers=1,
    )

    assert result.processed == 0


def test_per_folder_resume_reads_subfolder_manifest(tmp_path, dummy_detector):
    write_image(tmp_path / "sub" / "a.jpg")
    manifest_dir = tmp_path / "sub" / "_classified"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.csv").write_text(
        "source,destination,category,status,reason\n"
        f"{tmp_path / 'sub' / 'a.jpg'},,normal,copied,no_sensitive_label\n",
        encoding="utf-8",
    )

    result = classify_images.scan_and_classify(
        tmp_path,
        mode="copy",
        output_strategy="per-folder",
        batch_size=1,
        transfer_workers=1,
    )

    assert result.processed == 0


def test_batch_length_mismatch_falls_back_to_single_detection(tmp_path, monkeypatch):
    class BadBatchDetector(DummyDetector):
        def __init__(self) -> None:
            super().__init__()
            self.single_calls = 0

        def detect_batch(self, paths):
            return []

        def detect(self, path):
            self.single_calls += 1
            return []

    detector = BadBatchDetector()
    monkeypatch.setattr(
        classify_images,
        "load_detector",
        lambda *args, **kwargs: (detector, ["CPUExecutionProvider"]),
    )
    write_image(tmp_path / "a.jpg")
    write_image(tmp_path / "b.jpg")

    result = classify_images.scan_and_classify(
        tmp_path,
        mode="copy",
        output_strategy="root",
        batch_size=2,
        transfer_workers=1,
    )

    assert result.processed == 2
    assert result.batch_errors == 1
    assert detector.single_calls == 2
