from __future__ import annotations

import csv
import threading
from pathlib import Path

import pytest

import classify_images


def write_image(path: Path, content: bytes = b"fake image bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


class PipelinedDetector:
    """Fake detector exposing the prepare_batch/run_prepared API."""

    accepts_unicode_paths = True

    def __init__(self, scores=None, fail_run=False, on_run=None) -> None:
        self.scores = scores or {}
        self.fail_run = fail_run
        self.on_run = on_run
        self.events: list[tuple[str, tuple[str, ...]]] = []
        self.single_calls: list[Path] = []
        self.closed = False

    def prepare_batch(self, paths):
        paths = list(paths)
        self.events.append(("prepare", tuple(p.name for p in paths)))
        return paths

    def run_prepared(self, prepared):
        self.events.append(("run", tuple(p.name for p in prepared)))
        if self.on_run:
            self.on_run()
        if self.fail_run:
            raise RuntimeError("session exploded")
        results = []
        for path in prepared:
            if path.name.startswith("corrupt"):
                results.append(ValueError("cannot decode"))
            elif not path.exists():
                results.append(FileNotFoundError(path))
            else:
                results.append(self.scores.get(path.name, []))
        return results

    def detect(self, path):
        self.single_calls.append(Path(path))
        return self.scores.get(Path(path).name, [])

    def close(self) -> None:
        self.closed = True


def install(monkeypatch, detector):
    monkeypatch.setattr(
        classify_images,
        "load_detector",
        lambda *args, **kwargs: (detector, ["CPUExecutionProvider"]),
    )
    return detector


def run(root, **kwargs):
    options = dict(mode="copy", batch_size=2, transfer_workers=1)
    options.update(kwargs)
    return classify_images.scan_and_classify(root, **options)


def test_pipelined_detector_classifies_into_categories(tmp_path, monkeypatch):
    detector = install(
        monkeypatch,
        PipelinedDetector(
            scores={
                "n.jpg": [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.9}],
                "s.jpg": [{"class": "BELLY_EXPOSED", "score": 0.9}],
            }
        ),
    )
    for name in ("n.jpg", "s.jpg", "ok.jpg"):
        write_image(tmp_path / name)

    result = run(tmp_path, nude_threshold=0.5, sexy_threshold=0.5)

    assert result.processed == 3 and result.errors == 0
    assert (tmp_path / "_classified" / "nude" / "n.jpg").exists()
    assert (tmp_path / "_classified" / "sexy" / "s.jpg").exists()
    assert (tmp_path / "_classified" / "normal" / "ok.jpg").exists()
    assert detector.single_calls == []


def test_next_batch_is_prepared_before_current_batch_runs(tmp_path, monkeypatch):
    detector = install(monkeypatch, PipelinedDetector())
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"):
        write_image(tmp_path / name)

    result = run(tmp_path, batch_size=2)

    assert result.processed == 5
    assert detector.events == [
        ("prepare", ("a.jpg", "b.jpg")),
        ("prepare", ("c.jpg", "d.jpg")),
        ("run", ("a.jpg", "b.jpg")),
        ("prepare", ("e.jpg",)),
        ("run", ("c.jpg", "d.jpg")),
        ("run", ("e.jpg",)),
    ]


def test_bad_image_goes_to_errors_without_single_fallback(tmp_path, monkeypatch):
    detector = install(monkeypatch, PipelinedDetector())
    write_image(tmp_path / "a.jpg")
    write_image(tmp_path / "corrupt.jpg")
    write_image(tmp_path / "z.jpg")

    result = run(tmp_path, batch_size=3)

    assert result.processed == 2
    assert result.errors == 1
    assert result.batch_errors == 0
    assert detector.single_calls == []
    assert (tmp_path / "_classified" / "errors" / "corrupt.jpg").exists()
    rows = {Path(r["source"]).name: r for r in read_manifest(tmp_path / "_classified" / "manifest.csv")}
    assert rows["corrupt.jpg"]["category"] == "errors"
    assert "ValueError" in rows["corrupt.jpg"]["reason"]
    assert rows["a.jpg"]["status"] == "copied"
    assert result.log_path.exists()  # errors keep a debug log


def test_missing_file_is_recorded_as_skipped_missing(tmp_path, monkeypatch):
    install(monkeypatch, PipelinedDetector())
    present = write_image(tmp_path / "present.jpg")
    ghost = tmp_path / "ghost.jpg"
    # Detector sees the path but the file vanished before it was read.
    monkeypatch.setattr(
        classify_images,
        "iter_images",
        lambda root, output_dir: iter([present, ghost]),
    )

    result = run(tmp_path, batch_size=2)

    assert result.processed == 1
    assert result.skipped == 1
    rows = read_manifest(tmp_path / "_classified" / "manifest.csv")
    assert {r["status"] for r in rows} == {"copied", "skipped_missing"}
    # skipped_missing counts as done, so a rerun does not retry it.
    again = run(tmp_path, batch_size=2)
    assert again.processed == 0


def test_session_failure_falls_back_to_single_detection(tmp_path, monkeypatch):
    detector = install(monkeypatch, PipelinedDetector(fail_run=True))
    for name in ("a.jpg", "b.jpg"):
        write_image(tmp_path / name)

    result = run(tmp_path, batch_size=2)

    assert result.batch_errors == 1
    assert result.processed == 2
    assert sorted(p.name for p in detector.single_calls) == ["a.jpg", "b.jpg"]


def test_cancel_stops_pipelined_run_before_next_batch(tmp_path, monkeypatch):
    cancel_event = threading.Event()
    detector = install(
        monkeypatch, PipelinedDetector(on_run=cancel_event.set)
    )
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        write_image(tmp_path / name)

    result = run(tmp_path, batch_size=1, cancel_event=cancel_event)

    assert result.cancelled
    assert result.processed == 0
    assert [e for e in detector.events if e[0] == "run"] == [("run", ("a.jpg",))]


def test_cancelled_run_can_be_resumed(tmp_path, monkeypatch):
    cancel_event = threading.Event()
    install(monkeypatch, PipelinedDetector(on_run=cancel_event.set))
    for name in ("a.jpg", "b.jpg"):
        write_image(tmp_path / name)
    run(tmp_path, batch_size=1, cancel_event=cancel_event)

    install(monkeypatch, PipelinedDetector())
    result = run(tmp_path, batch_size=1)

    assert result.processed == 2


def test_progress_callback_reports_completion(tmp_path, monkeypatch):
    install(monkeypatch, PipelinedDetector())
    for index in range(5):
        write_image(tmp_path / f"img{index}.jpg")
    calls = []

    run(
        tmp_path,
        batch_size=2,
        progress=lambda done, total, path, category: calls.append((done, total)),
        progress_interval=100,
    )

    assert calls == [(5, 5)]


def test_duplicate_names_get_unique_destinations(tmp_path, monkeypatch):
    install(monkeypatch, PipelinedDetector())
    write_image(tmp_path / "one" / "same.jpg", b"1")
    write_image(tmp_path / "two" / "same.jpg", b"2")
    write_image(tmp_path / "three" / "same.jpg", b"3")

    result = run(tmp_path, batch_size=3)

    assert result.processed == 3
    names = sorted(p.name for p in (tmp_path / "_classified" / "normal").iterdir())
    assert names == ["same.jpg", "same_1.jpg", "same_2.jpg"]


def test_images_inside_output_folder_are_not_rescanned(tmp_path, monkeypatch):
    install(monkeypatch, PipelinedDetector())
    write_image(tmp_path / "a.jpg")

    run(tmp_path)
    second = run(tmp_path)

    assert second.total_seen == 1
    assert second.processed == 0


def test_move_mode_removes_sources(tmp_path, monkeypatch):
    install(monkeypatch, PipelinedDetector())
    source = write_image(tmp_path / "a.jpg")

    result = run(tmp_path, mode="move")

    assert result.processed == 1
    assert not source.exists()
    assert (tmp_path / "_classified" / "normal" / "a.jpg").exists()
    assert read_manifest(tmp_path / "_classified" / "manifest.csv")[0]["status"] == "moved"


def test_per_folder_run_without_errors_creates_no_root_output(tmp_path, monkeypatch):
    install(monkeypatch, PipelinedDetector())
    write_image(tmp_path / "sub" / "a.jpg")

    result = run(tmp_path, output_strategy="per-folder")

    assert result.processed == 1
    assert (tmp_path / "sub" / "_classified" / "normal" / "a.jpg").exists()
    assert not (tmp_path / "_classified").exists()


def test_per_folder_error_log_creates_directory_lazily(tmp_path, monkeypatch):
    install(monkeypatch, PipelinedDetector())
    write_image(tmp_path / "sub" / "corrupt.jpg")

    result = run(tmp_path, output_strategy="per-folder")

    assert result.errors == 1
    assert result.log_path.exists()
    assert (tmp_path / "sub" / "_classified" / "errors" / "corrupt.jpg").exists()


def test_injected_detector_is_not_closed_by_scan_and_classify(tmp_path):
    detector = PipelinedDetector()
    write_image(tmp_path / "a.jpg")

    result = classify_images.scan_and_classify(
        tmp_path,
        mode="copy",
        batch_size=1,
        transfer_workers=1,
        detector=detector,
        providers=["CPUExecutionProvider"],
    )

    assert result.processed == 1
    assert result.providers == ["CPUExecutionProvider"]
    assert not detector.closed


def test_scan_folders_loads_model_once_and_closes_it(tmp_path, monkeypatch):
    detector = PipelinedDetector()
    loads = []

    def fake_load(*args, **kwargs):
        loads.append(args)
        return detector, ["CPUExecutionProvider"]

    monkeypatch.setattr(classify_images, "load_detector", fake_load)
    folders = []
    for name in ("one", "two", "three"):
        folder = tmp_path / name
        write_image(folder / "a.jpg")
        folders.append(folder)
    seen = []

    results = classify_images.scan_folders(
        folders,
        mode="copy",
        batch_size=2,
        transfer_workers=1,
        on_folder=lambda index, total, folder: seen.append((index, total, folder.name)),
    )

    assert len(loads) == 1
    assert [r.processed for _, r in results] == [1, 1, 1]
    assert seen == [(1, 3, "one"), (2, 3, "two"), (3, 3, "three")]
    assert detector.closed


def test_scan_folders_stops_between_folders_when_cancelled(tmp_path, monkeypatch):
    cancel_event = threading.Event()
    detector = install(monkeypatch, PipelinedDetector(on_run=cancel_event.set))
    folders = []
    for name in ("one", "two"):
        folder = tmp_path / name
        write_image(folder / "a.jpg")
        folders.append(folder)

    results = classify_images.scan_folders(
        folders, mode="copy", batch_size=1, transfer_workers=1, cancel_event=cancel_event
    )

    assert len(results) == 1 and results[0][1].cancelled
    assert detector.closed
    assert not (tmp_path / "two" / "_classified").exists()


def test_scan_folders_validates_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        classify_images,
        "load_detector",
        lambda *a, **k: pytest.fail("model must not load for invalid options"),
    )

    with pytest.raises(ValueError):
        classify_images.scan_folders([tmp_path / "does-not-exist"])
    with pytest.raises(ValueError):
        classify_images.scan_folders([tmp_path], batch_size=0)


def test_scan_folders_closes_detector_on_failure(tmp_path, monkeypatch):
    detector = PipelinedDetector()
    install(monkeypatch, detector)
    write_image(tmp_path / "a.jpg")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(classify_images, "scan_and_classify", boom)

    with pytest.raises(RuntimeError):
        classify_images.scan_folders([tmp_path])
    assert detector.closed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "rename"},
        {"batch_size": 0},
        {"device": "tpu"},
        {"output_strategy": "nested"},
        {"transfer_workers": -1},
        {"preprocess_workers": 0},
    ],
)
def test_scan_and_classify_rejects_invalid_options(tmp_path, kwargs):
    with pytest.raises(ValueError):
        classify_images.scan_and_classify(tmp_path, **kwargs)


def test_output_dir_conflicts_with_per_folder(tmp_path):
    with pytest.raises(ValueError):
        classify_images.scan_and_classify(
            tmp_path, output_dir=tmp_path / "out", output_strategy="per-folder"
        )
