"""Extra coverage: CLI, providers, transfer failures, concurrency, multi-folder, logs, GUI."""
from __future__ import annotations

import csv
import threading
from pathlib import Path

import numpy as np
import pytest

import classify_images
import transfer
from classify_images import ScanResult
from transfer import TransferJob, TransferOutcome, run_transfer, transfer_file


def write_image(path: Path, content: bytes = b"fake image bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


class FakeDetector:
    """Minimal pipelined detector; optional hook runs before each batch."""

    def __init__(self, before_run=None) -> None:
        self.before_run = before_run
        self.closed = False

    def prepare_batch(self, paths):
        return list(paths)

    def run_prepared(self, prepared):
        if self.before_run:
            self.before_run(prepared)
        return [[] if p.exists() else FileNotFoundError(p) for p in prepared]

    def detect(self, path):
        return []

    def close(self):
        self.closed = True


def use_detector(monkeypatch, detector):
    monkeypatch.setattr(
        classify_images,
        "load_detector",
        lambda *a, **k: (detector, ["CPUExecutionProvider"]),
    )
    return detector


# --- CLI -------------------------------------------------------------------


def test_cli_passes_options_to_scan_folders_and_prints_summary(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_scan_folders(folders, **options):
        captured["folders"] = folders
        captured["options"] = options
        result = ScanResult(5, 4, 0, 1, 0, tmp_path / "debug.log", ["CPUExecutionProvider"], False)
        return [(folders[0], result)]

    monkeypatch.setattr(classify_images, "scan_folders", fake_scan_folders)

    code = classify_images.main(
        [str(tmp_path), "--mode", "move", "--device", "cpu", "--batch-size", "16",
         "--preprocess-workers", "3", "--fast-decode", "--output-strategy", "per-folder",
         "--nude-threshold", "0.7", "--limit", "10"]
    )

    options = captured["options"]
    assert code == 0
    assert captured["folders"] == [tmp_path]
    assert options["mode"] == "move" and options["device"] == "cpu"
    assert options["batch_size"] == 16 and options["preprocess_workers"] == 3
    assert options["fast_decode"] is True and options["limit"] == 10
    assert options["output_strategy"] == "per-folder"
    assert options["nude_threshold"] == pytest.approx(0.7)
    out = capsys.readouterr().out
    assert "seen=5" in out and "processed=4" in out and "errors=1" in out and "log=" in out


def test_cli_rejects_output_with_multiple_folders(tmp_path):
    with pytest.raises(SystemExit):
        classify_images.main([str(tmp_path), str(tmp_path), "--output", str(tmp_path / "o")])


def test_cli_rejects_output_with_per_folder(tmp_path):
    with pytest.raises(SystemExit):
        classify_images.main(
            [str(tmp_path), "--output", str(tmp_path / "o"), "--output-strategy", "per-folder"]
        )


def test_cli_still_accepts_legacy_engine_onnx_but_not_nudenet(tmp_path):
    assert classify_images.parse_args([str(tmp_path), "--engine", "onnx"]).engine == "onnx"
    with pytest.raises(SystemExit):
        classify_images.parse_args([str(tmp_path), "--engine", "nudenet"])


def test_cli_defaults(tmp_path):
    args = classify_images.parse_args([str(tmp_path)])

    assert args.mode == "copy" and args.device == "auto"
    assert args.batch_size == 128 and args.preprocess_workers == 4
    assert args.fast_decode is False and args.output_strategy == "root"


# --- providers and detector loading ----------------------------------------


@pytest.fixture()
def fake_ort(monkeypatch):
    ort = pytest.importorskip("onnxruntime")

    def set_available(*providers):
        monkeypatch.setattr(ort, "get_available_providers", lambda: list(providers))

    return set_available


def test_providers_cpu_never_uses_cuda(fake_ort):
    fake_ort("CUDAExecutionProvider", "CPUExecutionProvider")

    assert classify_images.get_onnx_providers("cpu") == ["CPUExecutionProvider"]


def test_providers_gpu_puts_cuda_first(fake_ort):
    fake_ort("CUDAExecutionProvider", "CPUExecutionProvider")

    assert classify_images.get_onnx_providers("gpu") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert classify_images.get_onnx_providers("auto")[0] == "CUDAExecutionProvider"


def test_providers_gpu_without_cuda_raises_but_auto_falls_back(fake_ort):
    fake_ort("CPUExecutionProvider")

    with pytest.raises(RuntimeError):
        classify_images.get_onnx_providers("gpu")
    assert classify_images.get_onnx_providers("auto") == ["CPUExecutionProvider"]


def test_load_detector_gpu_requested_but_not_active_closes_and_raises(monkeypatch):
    import fast_onnx_detector

    instances = []

    class CpuOnlyDetector:
        def __init__(self, providers, preprocess_workers, fast_decode):
            self.providers = ["CPUExecutionProvider"]
            self.closed = False
            instances.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(fast_onnx_detector, "FastOnnxNudeDetector", CpuOnlyDetector)
    monkeypatch.setattr(
        classify_images, "get_onnx_providers", lambda device: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        classify_images.load_detector("gpu", 2, False)
    assert instances[0].closed


def test_load_detector_forwards_options(monkeypatch):
    import fast_onnx_detector

    seen = {}

    class Recorder:
        def __init__(self, providers, preprocess_workers, fast_decode):
            seen.update(providers=providers, workers=preprocess_workers, fast=fast_decode)
            self.providers = providers

    monkeypatch.setattr(fast_onnx_detector, "FastOnnxNudeDetector", Recorder)

    detector, providers = classify_images.load_detector("cpu", 5, True)

    assert seen == {"providers": ["CPUExecutionProvider"], "workers": 5, "fast": True}
    assert providers == ["CPUExecutionProvider"] and isinstance(detector, Recorder)


# --- transfer failures -----------------------------------------------------


def test_transfer_missing_source_raises_and_leaves_no_part_file(tmp_path):
    destination = tmp_path / "out" / "x.jpg"

    with pytest.raises(FileNotFoundError):
        transfer_file(tmp_path / "missing.jpg", destination, "copy")

    assert not list(destination.parent.glob("*"))


def test_transfer_move_across_drives_copies_then_removes_source(tmp_path, monkeypatch):
    source = write_image(tmp_path / "a.jpg", b"payload")
    destination = tmp_path / "out" / "a.jpg"
    drives = iter(["C:", "D:"])
    monkeypatch.setattr(transfer.os.path, "splitdrive", lambda p: (next(drives), p))

    transfer_file(source, destination, "move")

    assert not source.exists()
    assert destination.read_bytes() == b"payload"
    assert not list(destination.parent.glob("*.part"))


def test_transfer_copy_failure_midway_keeps_source_and_cleans_part(tmp_path, monkeypatch):
    source = write_image(tmp_path / "a.jpg", b"payload")
    destination = tmp_path / "out" / "a.jpg"

    def exploding_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(transfer.os, "replace", exploding_replace)

    with pytest.raises(OSError):
        transfer_file(source, destination, "copy")

    assert source.read_bytes() == b"payload"
    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))


def test_run_transfer_reports_failure_instead_of_raising(tmp_path):
    job = TransferJob(tmp_path / "missing.jpg", tmp_path / "o" / "x.jpg", "normal", "copied", "r")

    outcome = run_transfer(job, "copy")

    assert isinstance(outcome, TransferOutcome)
    assert not outcome.ok and "FileNotFoundError" in outcome.error


def test_source_vanishing_before_transfer_is_recorded_as_error(tmp_path, monkeypatch):
    victim = write_image(tmp_path / "victim.jpg")
    write_image(tmp_path / "ok.jpg")

    class Vanishing(FakeDetector):
        def run_prepared(self, prepared):
            results = [[] for _ in prepared]
            victim.unlink()  # gone after detection, before the copy runs
            return results

    use_detector(monkeypatch, Vanishing())

    result = classify_images.scan_and_classify(
        tmp_path, mode="copy", batch_size=2, transfer_workers=1
    )

    assert result.errors == 1 and result.processed == 1
    rows = {Path(r["source"]).name: r for r in read_manifest(tmp_path / "_classified" / "manifest.csv")}
    assert rows["victim.jpg"]["status"] == "error" and rows["victim.jpg"]["destination"] == ""
    assert rows["ok.jpg"]["status"] == "copied"
    assert not list((tmp_path / "_classified").rglob("*.part"))


def test_failed_transfer_is_retried_on_next_run(tmp_path, monkeypatch):
    victim = write_image(tmp_path / "victim.jpg")
    calls = {"n": 0}

    def flaky(job, mode):
        calls["n"] += 1
        if calls["n"] == 1:
            return TransferOutcome(job=job, ok=False, error="OSError('busy')")
        return run_transfer(job, mode)

    monkeypatch.setattr(classify_images, "run_transfer", flaky)
    use_detector(monkeypatch, FakeDetector())

    first = classify_images.scan_and_classify(tmp_path, batch_size=1, transfer_workers=1)
    second = classify_images.scan_and_classify(tmp_path, batch_size=1, transfer_workers=1)

    assert first.errors == 1 and first.processed == 0
    assert second.processed == 1 and second.errors == 0
    assert victim.exists()  # copy mode never touches the source


# --- concurrency -----------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_many_files_with_many_transfer_workers_stay_consistent(tmp_path, monkeypatch, workers):
    use_detector(monkeypatch, FakeDetector())
    total = 0
    for folder in ("a", "b", "c"):
        for index in range(40):
            write_image(tmp_path / folder / f"img{index}.jpg", f"{folder}{index}".encode())
            total += 1

    result = classify_images.scan_and_classify(
        tmp_path, mode="copy", batch_size=7, transfer_workers=workers
    )

    out = tmp_path / "_classified"
    copied = [p for p in (out / "normal").iterdir()]
    assert result.processed == total and result.errors == 0
    assert len(copied) == total and len({p.name.casefold() for p in copied}) == total
    assert {p.read_bytes() for p in copied} == {
        f"{f}{i}".encode() for f in ("a", "b", "c") for i in range(40)
    }
    assert len(read_manifest(out / "manifest.csv")) == total
    assert not list(out.rglob("*.part"))


def test_progress_is_monotonic_with_concurrent_transfers(tmp_path, monkeypatch):
    use_detector(monkeypatch, FakeDetector())
    for index in range(60):
        write_image(tmp_path / f"i{index}.jpg")
    seen = []

    classify_images.scan_and_classify(
        tmp_path, batch_size=5, transfer_workers=4, progress_interval=1,
        progress=lambda done, total, path, category: seen.append(done),
    )

    assert seen == sorted(seen) and seen[-1] == 60 and len(set(seen)) == len(seen)


def test_cancel_mid_run_leaves_consistent_resumable_state(tmp_path, monkeypatch):
    cancel = threading.Event()
    batches = {"n": 0}

    def maybe_cancel(prepared):
        batches["n"] += 1
        if batches["n"] == 3:
            cancel.set()

    use_detector(monkeypatch, FakeDetector(before_run=maybe_cancel))
    for index in range(20):
        write_image(tmp_path / f"i{index:02d}.jpg")

    first = classify_images.scan_and_classify(
        tmp_path, batch_size=2, transfer_workers=3, cancel_event=cancel
    )
    use_detector(monkeypatch, FakeDetector())
    second = classify_images.scan_and_classify(tmp_path, batch_size=2, transfer_workers=3)

    assert first.cancelled
    assert 0 < first.processed < 20
    assert first.processed + second.processed == 20
    done = [Path(r["source"]).name for r in read_manifest(tmp_path / "_classified" / "manifest.csv")]
    assert sorted(done) == sorted(f"i{i:02d}.jpg" for i in range(20))  # none twice, none lost


# --- multi-folder ----------------------------------------------------------


def test_scan_folders_per_folder_strategy_writes_manifest_in_each_subfolder(tmp_path, monkeypatch):
    use_detector(monkeypatch, FakeDetector())
    roots = []
    for name in ("setA", "setB"):
        root = tmp_path / name
        write_image(root / "x" / "same.jpg")
        write_image(root / "y" / "same.jpg")
        roots.append(root)

    results = classify_images.scan_folders(
        roots, mode="copy", batch_size=2, transfer_workers=1, output_strategy="per-folder"
    )

    assert [r.processed for _, r in results] == [2, 2]
    for root in roots:
        for sub in ("x", "y"):
            assert (root / sub / "_classified" / "normal" / "same.jpg").exists()
            assert len(read_manifest(root / sub / "_classified" / "manifest.csv")) == 1
        assert not (root / "_classified").exists()


def test_scan_folders_keeps_results_separate_per_root(tmp_path, monkeypatch):
    use_detector(monkeypatch, FakeDetector())
    roots = []
    for name, count in (("one", 1), ("two", 3)):
        root = tmp_path / name
        for index in range(count):
            write_image(root / f"{index}.jpg")
        roots.append(root)

    results = classify_images.scan_folders(roots, batch_size=2, transfer_workers=1)

    assert [(r.total_seen, r.processed) for _, r in results] == [(1, 1), (3, 3)]
    assert len(list((roots[0] / "_classified" / "normal").iterdir())) == 1
    assert len(list((roots[1] / "_classified" / "normal").iterdir())) == 3


def test_scan_folders_with_empty_list_does_not_load_model(monkeypatch):
    monkeypatch.setattr(
        classify_images, "load_detector", lambda *a, **k: pytest.fail("must not load")
    )

    assert classify_images.scan_folders([]) == []


def test_limit_caps_images_seen(tmp_path, monkeypatch):
    use_detector(monkeypatch, FakeDetector())
    for index in range(6):
        write_image(tmp_path / f"{index}.jpg")

    result = classify_images.scan_and_classify(tmp_path, limit=3, batch_size=2, transfer_workers=1)

    assert result.total_seen == 3 and result.processed == 3


def test_empty_folder_is_a_clean_noop(tmp_path, monkeypatch):
    use_detector(monkeypatch, FakeDetector())

    result = classify_images.scan_and_classify(tmp_path, batch_size=4, transfer_workers=1)

    assert (result.total_seen, result.processed, result.errors) == (0, 0, 0)
    assert not result.cancelled


def test_custom_output_dir_is_used_and_not_rescanned(tmp_path, monkeypatch):
    use_detector(monkeypatch, FakeDetector())
    write_image(tmp_path / "a.jpg")
    out = tmp_path / "results"

    first = classify_images.scan_and_classify(tmp_path, output_dir=out, batch_size=1, transfer_workers=1)
    second = classify_images.scan_and_classify(tmp_path, output_dir=out, batch_size=1, transfer_workers=1)

    assert first.processed == 1 and (out / "normal" / "a.jpg").exists()
    assert second.total_seen == 1 and second.processed == 0


# --- logging ---------------------------------------------------------------


def test_debug_log_is_written_and_handlers_are_closed(tmp_path, monkeypatch):
    use_detector(monkeypatch, FakeDetector())
    write_image(tmp_path / "a.jpg")

    result = classify_images.scan_and_classify(
        tmp_path, debug_log=True, batch_size=1, transfer_workers=1
    )

    text = result.log_path.read_text(encoding="utf-8")
    assert "Started" in text and "Finished" in text
    import logging

    assert not logging.getLogger("phan_loai_image").handlers
    result.log_path.unlink()  # would fail on Windows if the handle were still open


def test_stale_log_is_removed_when_run_is_clean(tmp_path, monkeypatch):
    use_detector(monkeypatch, FakeDetector())
    write_image(tmp_path / "a.jpg")
    log = tmp_path / "_classified" / "debug.log"
    log.parent.mkdir()
    log.write_text("old failure", encoding="utf-8")

    result = classify_images.scan_and_classify(tmp_path, batch_size=1, transfer_workers=1)

    assert result.errors == 0 and not log.exists()


def test_custom_log_path_is_honoured(tmp_path, monkeypatch):
    use_detector(monkeypatch, FakeDetector())
    write_image(tmp_path / "a.jpg")
    log = tmp_path / "logs" / "run.log"

    result = classify_images.scan_and_classify(
        tmp_path, debug_log=True, log_path=log, batch_size=1, transfer_workers=1
    )

    assert result.log_path == log.resolve() and log.exists()


def test_chunked_splits_evenly_and_keeps_remainder():
    assert list(classify_images.chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(classify_images.chunked([], 3)) == []
    assert list(classify_images.chunked(iter(range(3)), 3)) == [[0, 1, 2]]


# --- real detector: image kinds and ordering --------------------------------

cv2 = pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def real_detector():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("nudenet")
    from fast_onnx_detector import FastOnnxNudeDetector

    detector = FastOnnxNudeDetector(providers=["CPUExecutionProvider"], preprocess_workers=8)
    yield detector
    detector.close()


def save(path, image, extension=None, params=None):
    ok, buffer = cv2.imencode(extension or path.suffix, image, params or [])
    assert ok
    buffer.tofile(str(path))
    return path


def test_real_detector_handles_gray_rgba_and_palette_like_images(tmp_path, real_detector):
    rng = np.random.default_rng(5)
    gray = save(tmp_path / "gray.png", rng.integers(0, 256, (90, 120), dtype=np.uint8))
    rgba = save(tmp_path / "rgba.png", rng.integers(0, 256, (80, 60, 4), dtype=np.uint8))
    tiny = save(tmp_path / "tiny.jpg", rng.integers(0, 256, (3, 5, 3), dtype=np.uint8))
    wide = save(tmp_path / "wide.jpg", rng.integers(0, 256, (40, 900, 3), dtype=np.uint8))

    results = real_detector.run_prepared(real_detector.prepare_batch([gray, rgba, tiny, wide]))

    assert all(isinstance(r, list) for r in results), results


def test_real_detector_preserves_input_order_under_many_workers(tmp_path, real_detector):
    rng = np.random.default_rng(9)
    paths = [
        save(tmp_path / f"{i}.jpg", rng.integers(0, 256, (100 + 7 * i, 140, 3), dtype=np.uint8))
        for i in range(20)
    ]

    batched = real_detector.detect_batch(paths)
    singles = [real_detector.detect(p) for p in paths]

    assert [[d["class"] for d in r] for r in batched] == [[d["class"] for d in r] for r in singles]


def test_jpeg_dimensions_handles_progressive_and_grayscale(tmp_path):
    from fast_onnx_detector import jpeg_dimensions

    rng = np.random.default_rng(1)
    color = rng.integers(0, 256, (70, 130, 3), dtype=np.uint8)
    gray = rng.integers(0, 256, (70, 130), dtype=np.uint8)
    progressive = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_PROGRESSIVE, 1])[1]
    gray_jpeg = cv2.imencode(".jpg", gray)[1]

    assert jpeg_dimensions(progressive) == (130, 70)
    assert jpeg_dimensions(gray_jpeg) == (130, 70)


def test_jpeg_dimensions_skips_leading_app_segments_and_fill_bytes():
    from fast_onnx_detector import jpeg_dimensions

    app0 = b"\xff\xe0" + (2 + 4).to_bytes(2, "big") + b"JFIF"
    sof = b"\xff\xc0\x00\x11\x08" + (321).to_bytes(2, "big") + (654).to_bytes(2, "big") + bytes(10)
    data = b"\xff\xd8" + b"\xff\xff" + app0 + sof

    assert jpeg_dimensions(data) == (654, 321)


def test_fast_decode_handles_progressive_large_jpeg(tmp_path):
    from fast_onnx_detector import FastOnnxNudeDetector

    gradient = np.tile(np.linspace(0, 255, 1500, dtype=np.uint8), (1000, 1))
    image = np.stack([gradient, gradient, gradient[::-1]], axis=2)
    path = save(tmp_path / "p.jpg", image, params=[cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
    detector = FastOnnxNudeDetector(
        providers=["CPUExecutionProvider"], preprocess_workers=1, fast_decode=True
    )
    try:
        blob, meta = detector._read_and_preprocess(path)
    finally:
        detector.close()

    assert blob.shape == (1, 3, 320, 320) and meta == (0, 500, 1500, 1000)


# --- GUI --------------------------------------------------------------------


@pytest.fixture()
def window(monkeypatch):
    tk = pytest.importorskip("tkinter")
    import app

    try:
        instance = app.ImageClassifierApp()
    except tk.TclError:
        pytest.skip("no display available")
    instance.withdraw()
    calls = {"info": [], "error": []}
    monkeypatch.setattr(app.messagebox, "showinfo", lambda *a: calls["info"].append(a))
    monkeypatch.setattr(app.messagebox, "showerror", lambda *a: calls["error"].append(a))
    instance.calls = calls
    yield instance
    instance.destroy()


def drain(window):
    events = []
    while not window.events.empty():
        events.append(window.events.get_nowait())
    return events


def test_gui_folder_list_dedupes_and_ignores_non_directories(window, tmp_path):
    (tmp_path / "f").mkdir()
    window._add_folder(tmp_path / "f")
    window._add_folder(tmp_path / "f")
    window._add_folder(tmp_path / "missing")
    window._add_folder(write_image(tmp_path / "file.jpg"))

    assert window.folders == [(tmp_path / "f").resolve()]
    assert window.folder_list.size() == 1


def test_gui_remove_selected_and_clear(window, tmp_path):
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
        window._add_folder(tmp_path / name)
    window.folder_list.selection_set(0)
    window.folder_list.selection_set(2)

    window._remove_selected()

    assert [p.name for p in window.folders] == ["b"]
    window._clear_folders()
    assert window.folders == [] and window.folder_list.size() == 0


@pytest.mark.parametrize(
    "preset, device, batch, workers, transfer",
    [("hdd", "gpu", 256, 4, 1), ("ssd", "gpu", 256, 8, 1), ("cpu", "cpu", 64, 4, 1)],
)
def test_gui_presets(window, preset, device, batch, workers, transfer):
    window._apply_preset(preset)

    assert window.device_var.get() == device
    assert window.batch_size_var.get() == batch
    assert window.preprocess_workers_var.get() == workers
    assert window.transfer_workers_var.get() == transfer


def test_gui_start_without_folders_shows_error_and_starts_nothing(window):
    window._start()

    assert window.calls["error"] and window.worker is None


def test_gui_worker_forwards_config_and_emits_events(window, tmp_path, monkeypatch):
    import app

    captured = {}
    result = ScanResult(3, 3, 0, 0, 0, tmp_path / "d.log", ["CPUExecutionProvider"], False)

    def fake_scan_folders(folders, on_folder, progress, **options):
        captured.update(options)
        on_folder(1, 1, folders[0])
        progress(3, 3, Path("x.jpg"), "normal")
        return [(folders[0], result)]

    monkeypatch.setattr(app, "scan_folders", fake_scan_folders)
    config = app.RunConfig("move", 32, 0.7, 0.6, "cpu", 2, 5, "per-folder", True)

    window._run_worker([tmp_path], config)

    assert captured["mode"] == "move" and captured["batch_size"] == 32
    assert captured["device"] == "cpu" and captured["fast_decode"] is True
    assert captured["output_strategy"] == "per-folder" and captured["preprocess_workers"] == 5
    kinds = [event for event, _ in drain(window)]
    assert kinds == ["folder", "progress", "done"]


def test_gui_worker_reports_fatal_errors(window, tmp_path, monkeypatch):
    import app

    def boom(*args, **kwargs):
        raise RuntimeError("no cuda")

    monkeypatch.setattr(app, "scan_folders", boom)
    config = app.RunConfig("copy", 8, 0.8, 0.8, "gpu", 0, 4, "root", False)

    window._run_worker([tmp_path], config)

    (event, payload), = drain(window)
    assert event == "error" and "no cuda" in str(payload)


@pytest.mark.parametrize("cancelled, prefix", [(False, "Done"), (True, "Stopped")])
def test_gui_poll_reports_done_and_stopped(window, tmp_path, cancelled, prefix):
    result = ScanResult(10, 8, 0, 2, 1, tmp_path / "d.log", [], cancelled)
    window.events.put(("done", [(tmp_path, result)]))

    window._poll_events()

    status = window.status_var.get()
    assert status.startswith(prefix) and "processed=8" in status and "errors=2" in status
    assert window.calls["info"] and "debug.log" in window.detail_var.get()
    assert str(window.start_button.cget("state")) == "normal"


def test_gui_poll_reports_error_and_progress(window):
    window.events.put(("progress", (5, 10, "a.jpg", "errors")))
    window.events.put(("error", RuntimeError("kaput")))

    window._poll_events()

    assert "kaput" in window.status_var.get()
    assert window.calls["error"]
