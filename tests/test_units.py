from __future__ import annotations

from pathlib import Path

import pytest

from manifest import ManifestWriter, read_done_manifest, read_done_manifests
from policy import classify_detection
from scanner import OUTPUT_DIR_NAME, is_image, iter_images
from transfer import DestinationAllocator


# --- policy ----------------------------------------------------------------


def det(label, score):
    return {"class": label, "score": score}


def test_classify_nude_wins_over_sexy():
    category, reason = classify_detection(
        [det("BELLY_EXPOSED", 0.95), det("FEMALE_BREAST_EXPOSED", 0.7)], 0.6, 0.6
    )

    assert category == "nude"
    assert reason == "FEMALE_BREAST_EXPOSED:0.700"


def test_classify_sexy_when_below_nude_threshold():
    category, reason = classify_detection(
        [det("FEMALE_BREAST_EXPOSED", 0.5), det("BELLY_EXPOSED", 0.8)], 0.6, 0.6
    )

    assert category == "sexy"
    assert reason.startswith("FEMALE_BREAST_EXPOSED:") or reason.startswith("BELLY_EXPOSED:")


def test_classify_normal_for_empty_or_irrelevant_labels():
    assert classify_detection([], 0.5, 0.5) == ("normal", "no_sensitive_label")
    assert classify_detection([det("FACE_FEMALE", 0.99)], 0.5, 0.5)[0] == "normal"


def test_classify_threshold_is_inclusive():
    assert classify_detection([det("ANUS_EXPOSED", 0.5)], 0.5, 0.5)[0] == "nude"
    assert classify_detection([det("ANUS_EXPOSED", 0.49)], 0.5, 0.5)[0] == "normal"


def test_classify_uses_highest_score_per_group():
    category, reason = classify_detection(
        [det("BUTTOCKS_EXPOSED", 0.4), det("ANUS_EXPOSED", 0.9)], 0.5, 0.5
    )

    assert (category, reason) == ("nude", "ANUS_EXPOSED:0.900")


# --- scanner ---------------------------------------------------------------


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


@pytest.mark.parametrize("name", ["a.JPG", "b.jpeg", "c.PnG", "d.webp", "e.heic"])
def test_is_image_accepts_known_extensions_case_insensitively(name):
    assert is_image(Path(name))


@pytest.mark.parametrize("name", ["a.txt", "b.csv", "noext", "c.jpg.part"])
def test_is_image_rejects_other_files(name):
    assert not is_image(Path(name))


def test_iter_images_is_sorted_and_recursive(tmp_path):
    for relative in ("b.jpg", "a.jpg", "z/c.png", "m/d.jpg", "notes.txt"):
        touch(tmp_path / relative)

    found = [p.relative_to(tmp_path).as_posix() for p in iter_images(tmp_path, tmp_path / OUTPUT_DIR_NAME)]

    assert found == ["a.jpg", "b.jpg", "m/d.jpg", "z/c.png"]


def test_iter_images_skips_output_folders_at_any_depth(tmp_path):
    touch(tmp_path / "a.jpg")
    touch(tmp_path / OUTPUT_DIR_NAME / "normal" / "done.jpg")
    touch(tmp_path / "sub" / OUTPUT_DIR_NAME / "normal" / "done.jpg")
    touch(tmp_path / "sub" / "b.jpg")

    found = {p.name for p in iter_images(tmp_path, tmp_path / OUTPUT_DIR_NAME)}

    assert found == {"a.jpg", "b.jpg"}


def test_iter_images_skips_custom_output_dir(tmp_path):
    touch(tmp_path / "a.jpg")
    touch(tmp_path / "custom_out" / "normal" / "done.jpg")

    found = {p.name for p in iter_images(tmp_path, tmp_path / "custom_out")}

    assert found == {"a.jpg"}


# --- manifest --------------------------------------------------------------


def test_manifest_round_trip_and_resume_statuses(tmp_path):
    path = tmp_path / "manifest.csv"
    with ManifestWriter(path) as writer:
        writer.append(Path("ok.jpg"), Path("out/ok.jpg"), "normal", "copied", "r")
        writer.append(Path("moved.jpg"), Path("out/m.jpg"), "sexy", "moved")
        writer.append(Path("gone.jpg"), None, "", "skipped_missing", "missing")
        writer.append(Path("bad.jpg"), None, "errors", "error", "boom")

    assert read_done_manifest(path) == {"ok.jpg", "moved.jpg", "gone.jpg"}


def test_manifest_appends_without_duplicating_header(tmp_path):
    path = tmp_path / "manifest.csv"
    for name in ("a.jpg", "b.jpg"):
        with ManifestWriter(path) as writer:
            writer.append(Path(name), Path("o"), "normal", "copied")

    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "source,destination,category,status,reason"
    assert len(lines) == 3


def test_manifest_flushes_periodically(tmp_path):
    path = tmp_path / "manifest.csv"
    with ManifestWriter(path, flush_every=2) as writer:
        writer.append(Path("a.jpg"), None, "normal", "copied")
        writer.append(Path("b.jpg"), None, "normal", "copied")

        assert "b.jpg" in path.read_text(encoding="utf-8")


def test_manifest_writer_requires_open():
    writer = ManifestWriter(Path("never.csv"))

    with pytest.raises(RuntimeError):
        writer.append(Path("a.jpg"), None, "normal", "copied")


def test_read_done_manifest_missing_file_is_empty(tmp_path):
    assert read_done_manifest(tmp_path / "nope.csv") == set()


def test_read_done_manifests_unions_files(tmp_path):
    first, second = tmp_path / "1.csv", tmp_path / "2.csv"
    for path, name in ((first, "a.jpg"), (second, "b.jpg")):
        with ManifestWriter(path) as writer:
            writer.append(Path(name), None, "normal", "copied")

    assert read_done_manifests([first, second, tmp_path / "none.csv"]) == {"a.jpg", "b.jpg"}


def test_manifest_handles_commas_quotes_and_newlines(tmp_path):
    path = tmp_path / "manifest.csv"
    with ManifestWriter(path) as writer:
        writer.append(Path('we,ird "name".jpg'), None, "errors", "error", "line1\nline2")
    # an error row is not "done"; make a done one with the same tricky name
    with ManifestWriter(path) as writer:
        writer.append(Path('we,ird "name".jpg'), Path("o"), "normal", "copied")

    assert read_done_manifest(path) == {'we,ird "name".jpg'}


# --- destination allocator -------------------------------------------------


def test_allocator_returns_plain_name_first_then_suffixes(tmp_path):
    allocator = DestinationAllocator()

    names = [allocator.reserve(tmp_path, Path("x/photo.jpg")).name for _ in range(3)]

    assert names == ["photo.jpg", "photo_1.jpg", "photo_2.jpg"]


def test_allocator_respects_existing_files_and_is_case_insensitive(tmp_path):
    (tmp_path / "Photo.JPG").write_bytes(b"x")
    (tmp_path / "photo_1.jpg").write_bytes(b"x")
    allocator = DestinationAllocator()

    assert allocator.reserve(tmp_path, Path("photo.jpg")).name == "photo_2.jpg"


def test_allocator_keeps_directories_separate(tmp_path):
    allocator = DestinationAllocator()
    (tmp_path / "a").mkdir()

    first = allocator.reserve(tmp_path / "a", Path("p.jpg"))
    second = allocator.reserve(tmp_path / "b", Path("p.jpg"))

    assert first.name == second.name == "p.jpg"
    assert first.parent != second.parent


def test_allocator_handles_missing_directory(tmp_path):
    allocator = DestinationAllocator()

    assert allocator.reserve(tmp_path / "new", Path("p.jpg")) == tmp_path / "new" / "p.jpg"


def test_allocator_does_not_touch_filesystem_after_first_listing(tmp_path, monkeypatch):
    allocator = DestinationAllocator()
    allocator.reserve(tmp_path, Path("a.jpg"))
    import transfer

    monkeypatch.setattr(
        transfer.os, "scandir", lambda *_: pytest.fail("directory listed twice")
    )

    assert allocator.reserve(tmp_path, Path("b.jpg")).name == "b.jpg"
