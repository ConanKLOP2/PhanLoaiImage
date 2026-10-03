from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

import evaluate
import label_tool
import make_sample
import score_images
from accuracy import read_labels, read_sample, read_scores
from label_tool import LabelSession


def write_image(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class ScoringDetector:
    """Fake pipelined detector: file names decide the detections."""

    def __init__(self) -> None:
        self.closed = False
        self.batches: list[list[str]] = []

    def prepare_batch(self, paths):
        return list(paths)

    def run_prepared(self, prepared):
        self.batches.append([p.name for p in prepared])
        results = []
        for path in prepared:
            name = path.name
            if name.startswith("bad"):
                results.append(ValueError("cannot decode"))
            elif name.startswith("nude"):
                results.append([{"class": "ANUS_EXPOSED", "score": 0.9, "box": [1, 2, 3, 4]}])
            elif name.startswith("sexy"):
                results.append([{"class": "BELLY_EXPOSED", "score": 0.8, "box": [1, 2, 3, 4]}])
            else:
                results.append([])
        return results

    def close(self):
        self.closed = True


class BatchOnlyDetector:
    """Detector without the pipelined API."""

    def detect_batch(self, paths):
        return [[] for _ in paths]

    def detect(self, path):
        return []


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- score_images -----------------------------------------------------------


def test_score_folders_writes_one_line_per_image(tmp_path):
    for name in ("nude1.jpg", "sexy1.jpg", "plain1.jpg"):
        write_image(tmp_path / "set" / name)
    out = tmp_path / "scores.jsonl"

    counts = score_images.score_folders([tmp_path / "set"], out, ScoringDetector(), batch_size=2)

    written = {Path(r["source"]).name: r for r in rows(out)}
    assert counts == {"scored": 3, "errors": 0, "skipped": 0}
    assert written["nude1.jpg"]["detections"][0] == {"class": "ANUS_EXPOSED", "score": 0.9, "box": [1, 2, 3, 4]}
    assert written["plain1.jpg"]["detections"] == [] and written["plain1.jpg"]["error"] is None


def test_score_folders_records_errors_without_losing_the_batch(tmp_path):
    for name in ("bad1.jpg", "nude1.jpg"):
        write_image(tmp_path / name)
    out = tmp_path / "scores.jsonl"

    counts = score_images.score_folders([tmp_path], out, ScoringDetector(), batch_size=8)

    written = {Path(r["source"]).name: r for r in rows(out)}
    assert counts["errors"] == 1 and counts["scored"] == 1
    assert "ValueError" in written["bad1.jpg"]["error"]
    assert written["nude1.jpg"]["error"] is None
    assert list(read_scores(out)) == [str(tmp_path / "nude1.jpg")]


def test_score_folders_resumes_and_skips_done_images(tmp_path):
    write_image(tmp_path / "a.jpg")
    out = tmp_path / "scores.jsonl"
    score_images.score_folders([tmp_path], out, ScoringDetector())
    write_image(tmp_path / "b.jpg")
    detector = ScoringDetector()

    counts = score_images.score_folders([tmp_path], out, detector)

    assert counts == {"scored": 1, "errors": 0, "skipped": 1}
    assert detector.batches == [["b.jpg"]]
    assert len(rows(out)) == 2


def test_score_folders_tolerates_half_written_last_line(tmp_path):
    write_image(tmp_path / "a.jpg")
    out = tmp_path / "scores.jsonl"
    out.write_text('{"source": "x", "detections": [], "error": null}\n{"source": "tru', encoding="utf-8")

    assert score_images.read_done(out) == {"x"}


def test_score_folders_ignores_output_folder_and_non_images(tmp_path):
    write_image(tmp_path / "a.jpg")
    write_image(tmp_path / "_classified" / "normal" / "done.jpg")
    write_image(tmp_path / "notes.txt")
    out = tmp_path / "scores.jsonl"

    counts = score_images.score_folders([tmp_path], out, ScoringDetector())

    assert counts["scored"] == 1


def test_score_folders_limit_and_multiple_roots(tmp_path):
    for root in ("one", "two"):
        for index in range(3):
            write_image(tmp_path / root / f"{index}.jpg")
    out = tmp_path / "scores.jsonl"

    counts = score_images.score_folders(
        [tmp_path / "one", tmp_path / "two"], out, ScoringDetector(), limit=4
    )

    assert counts["scored"] == 4


def test_score_folders_works_with_non_pipelined_detector(tmp_path):
    write_image(tmp_path / "a.jpg")
    out = tmp_path / "scores.jsonl"

    counts = score_images.score_folders([tmp_path], out, BatchOnlyDetector())

    assert counts["scored"] == 1


def test_score_folders_falls_back_to_single_detection_when_batch_fails(tmp_path):
    class Exploding(BatchOnlyDetector):
        def detect_batch(self, paths):
            raise RuntimeError("session died")

        def detect(self, path):
            if "bad" in Path(path).name:
                raise ValueError("no")
            return []

    for name in ("a.jpg", "bad.jpg"):
        write_image(tmp_path / name)
    out = tmp_path / "scores.jsonl"

    counts = score_images.score_folders([tmp_path], out, Exploding(), batch_size=4)

    assert counts["scored"] == 1 and counts["errors"] == 1


def test_score_folders_stops_when_cancelled(tmp_path):
    for index in range(4):
        write_image(tmp_path / f"{index}.jpg")
    cancel = threading.Event()
    cancel.set()
    out = tmp_path / "scores.jsonl"

    counts = score_images.score_folders([tmp_path], out, ScoringDetector(), cancel_event=cancel)

    assert counts["scored"] == 0


def test_score_images_main_closes_detector_and_reports(tmp_path, monkeypatch, capsys):
    write_image(tmp_path / "a.jpg")
    detector = ScoringDetector()
    monkeypatch.setattr(score_images, "load_detector", lambda *a, **k: (detector, ["CPUExecutionProvider"]))
    out = tmp_path / "s.jsonl"

    code = score_images.main([str(tmp_path), "--out", str(out), "--device", "cpu"])

    assert code == 0 and detector.closed and out.exists()
    assert "scored=1" in capsys.readouterr().out


def test_score_images_main_rejects_missing_folder(tmp_path):
    with pytest.raises(SystemExit):
        score_images.main([str(tmp_path / "nope")])


# --- LabelSession -----------------------------------------------------------


def test_label_session_walks_through_images_and_saves_each_label(tmp_path):
    session = LabelSession(["a", "b", "c"], tmp_path / "labels.csv")

    assert session.current == "a"
    session.set_label("nude")
    session.set_label("normal")

    assert session.current == "c" and session.labelled_count == 2
    assert read_labels(tmp_path / "labels.csv") == {"a": "nude", "b": "normal"}
    assert session.counts() == {"nude": 1, "sexy": 0, "normal": 1}


def test_label_session_resumes_from_existing_file(tmp_path):
    path = tmp_path / "labels.csv"
    first = LabelSession(["a", "b", "c"], path)
    first.set_label("sexy")

    second = LabelSession(["a", "b", "c"], path)

    assert second.current == "b" and second.labelled_count == 1


def test_label_session_undo_reverts_label_and_file(tmp_path):
    path = tmp_path / "labels.csv"
    session = LabelSession(["a", "b"], path)
    session.set_label("nude")

    reverted = session.undo()

    assert reverted == "a" and session.current == "a"
    assert read_labels(path) == {}


def test_label_session_skip_and_undo_skip(tmp_path):
    session = LabelSession(["a", "b"], tmp_path / "labels.csv")

    session.skip()
    assert session.current == "b"
    session.undo()

    assert session.current == "a"


def test_label_session_skipped_images_are_not_saved(tmp_path):
    path = tmp_path / "labels.csv"
    session = LabelSession(["a", "b"], path)
    session.skip()
    session.set_label("normal")

    assert read_labels(path) == {"b": "normal"}
    assert session.current is None


def test_label_session_rejects_unknown_label_and_ignores_finished(tmp_path):
    session = LabelSession(["a"], tmp_path / "labels.csv")

    with pytest.raises(ValueError):
        session.set_label("maybe")
    session.set_label("sexy")
    session.set_label("nude")  # nothing left to label: no-op

    assert read_labels(tmp_path / "labels.csv") == {"a": "sexy"}
    assert session.current is None
    assert session.undo() == "a"  # undo still works after finishing


def test_label_session_undo_with_empty_history_is_safe(tmp_path):
    session = LabelSession(["a"], tmp_path / "labels.csv")

    assert session.undo() == "a"


def test_label_session_save_leaves_no_temp_file(tmp_path):
    session = LabelSession(["a", "b"], tmp_path / "labels.csv")
    session.set_label("nude")
    session.set_label("sexy")

    assert sorted(p.name for p in tmp_path.iterdir()) == ["labels.csv"]


def test_label_session_handles_awkward_paths(tmp_path):
    awkward = 'C:\\x, "y"\\ảnh 1.jpg'
    path = tmp_path / "labels.csv"
    LabelSession([awkward], path).set_label("normal")

    assert read_labels(path) == {awkward: "normal"}


def test_key_bindings_cover_all_classes():
    assert set(label_tool.KEY_TO_LABEL.values()) == {"nude", "sexy", "normal"}


def test_render_png_shrinks_large_jpeg(tmp_path):
    cv2 = pytest.importorskip("cv2")
    gradient = np.tile(np.linspace(0, 255, 3000, dtype=np.uint8), (2000, 1))
    path = tmp_path / "big.jpg"
    cv2.imencode(".jpg", np.stack([gradient] * 3, axis=2))[1].tofile(str(path))

    png = label_tool.render_png(str(path), 900, 700)
    decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)

    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert decoded.shape[1] <= 900 and decoded.shape[0] <= 700


def test_render_png_keeps_small_image_size_and_rejects_garbage(tmp_path):
    cv2 = pytest.importorskip("cv2")
    small = tmp_path / "small.png"
    cv2.imencode(".png", np.zeros((50, 80, 3), np.uint8))[1].tofile(str(small))
    garbage = write_image(tmp_path / "garbage.jpg", b"not an image")

    decoded = cv2.imdecode(np.frombuffer(label_tool.render_png(str(small)), np.uint8), cv2.IMREAD_COLOR)

    assert decoded.shape[:2] == (50, 80)
    with pytest.raises(ValueError):
        label_tool.render_png(str(garbage))


# --- make_sample / evaluate CLIs ---------------------------------------------


def make_scores(tmp_path, normal=20, nude=8, sexy=6):
    path = tmp_path / "scores.jsonl"
    lines = []
    for index in range(normal):
        lines.append({"source": f"n{index}.jpg", "detections": [], "error": None})
    for index in range(nude):
        lines.append({"source": f"x{index}.jpg", "detections": [{"class": "ANUS_EXPOSED", "score": 0.6 + 0.04 * index}], "error": None})
    for index in range(sexy):
        lines.append({"source": f"s{index}.jpg", "detections": [{"class": "BELLY_EXPOSED", "score": 0.6 + 0.05 * index}], "error": None})
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
    return path


def truth(source):
    return {"x": "nude", "s": "sexy"}.get(source[0], "normal")


def test_make_sample_cli_writes_sample_and_prints_strata(tmp_path, capsys):
    scores = make_scores(tmp_path)
    out = tmp_path / "sample.csv"

    code = make_sample.main(["--scores", str(scores), "--per-class", "5", "--out", str(out)])

    sample = read_sample(out)
    assert code == 0 and len(sample) == 15
    text = capsys.readouterr().out
    assert "34 anh" in text and "nude" in text and "sexy" in text


def test_make_sample_cli_rejects_empty_scores(tmp_path):
    empty = tmp_path / "scores.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit):
        make_sample.main(["--scores", str(empty)])


def prepare_eval_files(tmp_path):
    scores = make_scores(tmp_path)
    sample_path = tmp_path / "sample.csv"
    make_sample.main(["--scores", str(scores), "--per-class", "5", "--out", str(sample_path)])
    labels_path = tmp_path / "labels.csv"
    session = LabelSession([row.path for row in read_sample(sample_path)], labels_path)
    while session.current:
        session.set_label(truth(session.current))
    return scores, sample_path, labels_path


def test_evaluate_cli_full_report(tmp_path, capsys):
    scores, sample, labels = prepare_eval_files(tmp_path)
    report = tmp_path / "report.json"

    code = evaluate.main(
        ["--scores", str(scores), "--labels", str(labels), "--sample", str(sample),
         "--folds", "3", "--errors", "5", "--out", str(report)]
    )

    text = capsys.readouterr().out
    assert code == 0
    for expected in ("Hien tai: nude>=0.55", "Hien tai: nude>=0.8", "Tot nhat", "Kiem tra cheo 3 folds", "confusion"):
        assert expected in text
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["weighted"] is True and data["labelled"] == 15
    assert 0.0 <= data["best"]["macro_f1"] <= 1.0
    assert data["best"]["objective"] >= max(b["objective"] for b in data["baselines"]) - 1e-9


def test_evaluate_cli_cost_objective_and_custom_baseline(tmp_path, capsys):
    scores, _, labels = prepare_eval_files(tmp_path)

    evaluate.main(
        ["--scores", str(scores), "--labels", str(labels), "--objective", "cost",
         "--miss-cost", "5", "--current", "0.6", "0.6", "--folds", "0"]
    )

    text = capsys.readouterr().out
    assert "Hien tai: nude>=0.6 sexy>=0.6" in text and "Hien tai: nude>=0.8" not in text
    assert "Kiem tra cheo" not in text


def test_evaluate_cli_skips_cross_validation_for_tiny_label_sets(tmp_path, capsys):
    scores = make_scores(tmp_path)
    labels = tmp_path / "labels.csv"
    labels.write_text("path,label\nn0.jpg,normal\nx0.jpg,nude\n", encoding="utf-8")

    evaluate.main(["--scores", str(scores), "--labels", str(labels), "--folds", "5"])

    assert "Bo qua kiem tra cheo" in capsys.readouterr().out


def test_evaluate_cli_errors_without_overlap(tmp_path):
    scores = make_scores(tmp_path)
    labels = tmp_path / "labels.csv"
    labels.write_text("path,label\nunknown.jpg,nude\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        evaluate.main(["--scores", str(scores), "--labels", str(labels)])


def test_format_metrics_contains_every_class_and_matrix(tmp_path):
    from accuracy import Item, evaluate as run_evaluate

    text = evaluate.format_metrics(run_evaluate([Item("a", "nude", [])], 0.5, 0.5), "Title")

    assert text.startswith("Title") and all(name in text for name in ("nude", "sexy", "normal"))


# --- the whole workflow ------------------------------------------------------


def test_end_to_end_score_sample_label_evaluate(tmp_path, capsys):
    folder = tmp_path / "images"
    for index in range(12):
        write_image(folder / f"plain{index}.jpg")
    for index in range(6):
        write_image(folder / f"nude{index}.jpg")
    for index in range(4):
        write_image(folder / f"sexy{index}.jpg")
    scores = tmp_path / "scores.jsonl"

    score_images.score_folders([folder], scores, ScoringDetector(), batch_size=5)
    sample = tmp_path / "sample.csv"
    make_sample.main(["--scores", str(scores), "--per-class", "4", "--out", str(sample)])
    labels = tmp_path / "labels.csv"
    session = LabelSession([r.path for r in read_sample(sample)], labels)
    while session.current:
        session.set_label({"n": "nude", "s": "sexy"}.get(Path(session.current).name[0], "normal"))
    capsys.readouterr()

    evaluate.main(["--scores", str(scores), "--labels", str(labels), "--sample", str(sample), "--folds", "0"])

    out = capsys.readouterr().out
    # The fake model is perfect at the default thresholds, so accuracy is 100%.
    assert "accuracy=100.0%" in out


def test_evaluate_cli_warns_when_a_class_has_no_labelled_images(tmp_path, capsys):
    scores = make_scores(tmp_path)
    labels = tmp_path / "labels.csv"
    rows_text = "path,label\n" + "".join(f"n{i}.jpg,normal\n" for i in range(6)) + "s0.jpg,sexy\ns1.jpg,sexy\n"
    labels.write_text(rows_text, encoding="utf-8")
    report = tmp_path / "r.json"

    evaluate.main(["--scores", str(scores), "--labels", str(labels), "--folds", "0", "--out", str(report)])

    assert "CANH BAO" in capsys.readouterr().out
    assert json.loads(report.read_text(encoding="utf-8"))["unmeasured_classes"] == ["nude"]


def test_evaluate_cli_has_no_warning_when_all_classes_are_labelled(tmp_path, capsys):
    scores, sample, labels = prepare_eval_files(tmp_path)

    evaluate.main(["--scores", str(scores), "--labels", str(labels), "--sample", str(sample), "--folds", "0"])

    assert "CANH BAO" not in capsys.readouterr().out
