from __future__ import annotations

import argparse
import csv
import inspect
import logging
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total: int | None = None, unit: str = "", desc: str = ""):
            self.total = total or 0
            self.count = 0
            self.desc = desc
            self.unit = unit

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            if self.total:
                print(f"{self.desc}: {self.count}/{self.total} {self.unit}")

        def update(self, value: int) -> None:
            self.count += value

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}

NUDE_LABELS = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

SEXY_LABELS = {
    "FEMALE_GENITALIA_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
    "BELLY_EXPOSED",
    "ARMPITS_EXPOSED",
    "MALE_BREAST_EXPOSED",
}

OUTPUT_DIR_NAME = "_classified"
MANIFEST_NAME = "manifest.csv"


@dataclass(frozen=True)
class ScanResult:
    total_seen: int
    processed: int
    skipped: int
    errors: int
    log_path: Path


def get_onnx_providers(device: str) -> list[str]:
    providers = ["CPUExecutionProvider"]
    if device == "cpu":
        return providers

    try:
        import onnxruntime as ort
    except ImportError:
        return providers

    available = ort.get_available_providers()
    if device in {"auto", "gpu"} and "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")
    elif device == "gpu":
        raise RuntimeError(
            "Khong thay CUDAExecutionProvider. Hay cai onnxruntime-gpu va CUDA/cuDNN phu hop."
        )
    return providers


def load_detector(device: str = "auto"):
    try:
        from nudenet import NudeDetector
    except ImportError as exc:
        raise RuntimeError(
            "Chua cai nudenet. Hay chay: pip install -r requirements.txt"
        ) from exc

    providers = get_onnx_providers(device)
    signature = inspect.signature(NudeDetector)
    kwargs = {}
    if "providers" in signature.parameters:
        kwargs["providers"] = providers
    elif "provider" in signature.parameters:
        kwargs["provider"] = providers[0]

    try:
        return NudeDetector(**kwargs), providers
    except TypeError:
        return NudeDetector(), providers


def setup_logger(log_path: Path, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("phan_loai_image")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8", delay=True)
    file_handler.setLevel(logging.DEBUG if debug else logging.ERROR)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class ManifestWriter:
    def __init__(self, manifest_path: Path, flush_every: int = 100) -> None:
        self.manifest_path = manifest_path
        self.flush_every = flush_every
        self.count = 0
        self.file = None
        self.writer = None

    def __enter__(self) -> "ManifestWriter":
        is_new = not self.manifest_path.exists()
        self.file = self.manifest_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=["source", "destination", "category", "status", "reason"],
        )
        if is_new:
            self.writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, traceback_obj) -> None:
        if self.file:
            self.file.flush()
            self.file.close()

    def append(
        self,
        source: Path,
        destination: Path | None,
        category: str,
        status: str,
        reason: str = "",
    ) -> None:
        if not self.writer or not self.file:
            raise RuntimeError("ManifestWriter is not open")

        self.writer.writerow(
            {
                "source": str(source),
                "destination": str(destination) if destination else "",
                "category": category,
                "status": status,
                "reason": reason,
            }
        )
        self.count += 1
        if self.count % self.flush_every == 0:
            self.file.flush()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images(root: Path, output_dir: Path) -> Iterable[Path]:
    output_dir = output_dir.resolve()
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if (current_path / dirname).resolve() != output_dir
        ]
        for filename in filenames:
            path = current_path / filename
            if is_image(path):
                yield path


def count_images(root: Path, output_dir: Path, limit: int | None = None) -> int:
    count = 0
    for _ in iter_images(root, output_dir):
        count += 1
        if limit is not None and count >= limit:
            return count
    return count


def read_done_manifest(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()

    done: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            source = row.get("source")
            status = row.get("status")
            if source and status in {"moved", "copied", "skipped_missing"}:
                done.add(source)
    return done


def append_manifest(
    manifest_path: Path,
    source: Path,
    destination: Path | None,
    category: str,
    status: str,
    reason: str = "",
) -> None:
    is_new = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["source", "destination", "category", "status", "reason"],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "source": str(source),
                "destination": str(destination) if destination else "",
                "category": category,
                "status": status,
                "reason": reason,
            }
        )


def classify_detection(
    detections: list[dict],
    nude_threshold: float,
    sexy_threshold: float,
) -> tuple[str, str]:
    best_nude = 0.0
    best_sexy = 0.0
    best_label = ""

    for item in detections:
        label = str(item.get("class", ""))
        score = float(item.get("score", 0.0))
        if label in NUDE_LABELS and score > best_nude:
            best_nude = score
            best_label = label
        if label in SEXY_LABELS and score > best_sexy:
            best_sexy = score
            if not best_label:
                best_label = label

    if best_nude >= nude_threshold:
        return "nude", f"{best_label}:{best_nude:.3f}"
    if best_sexy >= sexy_threshold:
        return "sexy", f"{best_label}:{best_sexy:.3f}"
    return "normal", "no_sensitive_label"


def unique_destination(destination_dir: Path, source: Path) -> Path:
    destination = destination_dir / source.name
    if not destination.exists():
        return destination

    stem = source.stem
    suffix = source.suffix
    index = 1
    while True:
        candidate = destination_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def transfer_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        shutil.move(str(source), str(destination))


def chunked(items: Iterable[Path], size: int) -> Iterable[list[Path]]:
    batch: list[Path] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def scan_and_classify(
    root: Path,
    output_dir: Path | None = None,
    mode: str = "move",
    batch_size: int = 16,
    nude_threshold: float = 0.55,
    sexy_threshold: float = 0.55,
    limit: int | None = None,
    progress: Callable[[int, int, Path, str], None] | None = None,
    log_path: Path | None = None,
    debug_log: bool = False,
    progress_interval: int = 25,
    device: str = "auto",
) -> ScanResult:
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Thu muc khong ton tai: {root}")
    if mode not in {"move", "copy"}:
        raise ValueError("mode phai la 'move' hoac 'copy'")
    if batch_size < 1:
        raise ValueError("batch_size phai >= 1")
    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device phai la 'auto', 'cpu', hoac 'gpu'")
    if progress_interval < 1:
        progress_interval = 1

    output_dir = (output_dir or root / OUTPUT_DIR_NAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    log_path = (log_path or output_dir / "debug.log").resolve()
    logger = setup_logger(log_path, debug=debug_log)

    done_sources = read_done_manifest(manifest_path)

    for category in ("nude", "sexy", "normal", "errors"):
        (output_dir / category).mkdir(parents=True, exist_ok=True)

    detector, providers = load_detector(device)
    if debug_log:
        logger.debug(
            "Started root=%s output_dir=%s mode=%s batch_size=%s limit=%s device=%s providers=%s",
            root,
            output_dir,
            mode,
            batch_size,
            limit,
            device,
            providers,
        )
    all_images: list[Path] = []
    for path in iter_images(root, output_dir):
        if limit is not None and len(all_images) >= limit:
            break
        all_images.append(path)

    pending_paths = [path for path in all_images if str(path) not in done_sources]
    total_pending = len(pending_paths)
    if debug_log:
        logger.debug(
            "images_found=%s pending=%s already_done=%s",
            len(all_images),
            total_pending,
            len(all_images) - total_pending,
        )
    processed = 0
    skipped = 0
    errors = 0
    completed = 0

    def maybe_report(done: int, total: int, path: Path, category: str) -> None:
        if not progress:
            return
        if done == total or done % progress_interval == 0 or category == "errors":
            progress(done, total, path, category)

    with ManifestWriter(manifest_path) as manifest, tqdm(
        total=total_pending, unit="img", desc="Classifying"
    ) as bar:
        for batch in chunked(pending_paths, batch_size):
            existing_batch = []
            for path in batch:
                if path.exists():
                    existing_batch.append(path)
                else:
                    manifest.append(path, None, "", "skipped_missing", "missing")
                    skipped += 1
                    completed += 1
                    bar.update(1)

            if not existing_batch:
                continue

            try:
                predictions = detector.detect_batch([str(path) for path in existing_batch])
            except Exception as exc:
                logger.exception(
                    "Batch failed. Falling back to single-file detection. batch_size=%s first_file=%s error=%r",
                    len(existing_batch),
                    existing_batch[0] if existing_batch else "",
                    exc,
                )
                predictions = []
                for path in existing_batch:
                    try:
                        predictions.append(detector.detect(str(path)))
                    except Exception as single_exc:
                        logger.exception(
                            "Single-file detection failed: file=%s error=%r",
                            path,
                            single_exc,
                        )
                        destination = unique_destination(output_dir / "errors", path)
                        try:
                            transfer_file(path, destination, mode)
                            if debug_log:
                                logger.debug(
                                    "Moved errored file: source=%s destination=%s",
                                    path,
                                    destination,
                                )
                        except Exception as transfer_exc:
                            logger.exception(
                                "Could not move/copy errored file: file=%s error=%r",
                                path,
                                transfer_exc,
                            )
                            destination = None
                        error_reason = (
                            f"{type(single_exc).__name__}: {single_exc}\n"
                            f"{traceback.format_exc()}"
                        )
                        manifest.append(
                            path,
                            destination,
                            "errors",
                            "error",
                            error_reason,
                        )
                        errors += 1
                        completed += 1
                        bar.update(1)
                        maybe_report(completed, total_pending, path, "errors")
                        continue

                    category, reason = classify_detection(
                        predictions[-1], nude_threshold, sexy_threshold
                    )
                    destination = unique_destination(output_dir / category, path)
                    try:
                        transfer_file(path, destination, mode)
                        manifest.append(
                            path,
                            destination,
                            category,
                            f"{mode}d",
                            reason,
                        )
                        logger.debug(
                            "Classified after fallback: source=%s category=%s reason=%s destination=%s",
                            path,
                            category,
                            reason,
                            destination,
                        )
                        processed += 1
                    except Exception as transfer_exc:
                        logger.exception(
                            "Transfer failed after fallback: file=%s category=%s error=%r",
                            path,
                            category,
                            transfer_exc,
                        )
                        destination = None
                        manifest.append(
                            path,
                            None,
                            category,
                            "error",
                            f"{type(transfer_exc).__name__}: {transfer_exc}",
                        )
                        errors += 1
                    completed += 1
                    bar.update(1)
                    maybe_report(completed, total_pending, path, category)
                continue

            for path, detections in zip(existing_batch, predictions):
                category, reason = classify_detection(
                    detections, nude_threshold, sexy_threshold
                )
                destination = unique_destination(output_dir / category, path)
                try:
                    transfer_file(path, destination, mode)
                    manifest.append(path, destination, category, f"{mode}d", reason)
                    if debug_log:
                        logger.debug(
                            "Classified: source=%s category=%s reason=%s destination=%s detections=%s",
                            path,
                            category,
                            reason,
                            destination,
                            detections,
                        )
                    processed += 1
                    maybe_report(completed + 1, total_pending, path, category)
                except Exception as exc:
                    logger.exception(
                        "Transfer failed: file=%s category=%s error=%r",
                        path,
                        category,
                        exc,
                    )
                    manifest.append(
                        path,
                        None,
                        category,
                        "error",
                        repr(exc),
                    )
                    errors += 1
                completed += 1
                bar.update(1)

    if debug_log:
        logger.debug(
            "Finished seen=%s processed=%s skipped=%s errors=%s log=%s",
            len(all_images),
            processed,
            skipped,
            errors,
            log_path,
        )
    return ScanResult(
        total_seen=len(all_images),
        processed=processed,
        skipped=skipped,
        errors=errors,
        log_path=log_path,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phan loai anh thanh nude, sexy, normal bang NudeNet."
    )
    parser.add_argument("folder", type=Path, help="Thu muc chua anh can phan loai")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Thu muc output. Mac dinh: <folder>\\_classified",
    )
    parser.add_argument(
        "--mode",
        choices=["move", "copy"],
        default="move",
        help="move de chuyen file, copy de giu file goc",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="auto dung GPU neu onnxruntime CUDA kha dung, gpu bat buoc CUDA, cpu bat buoc CPU.",
    )
    parser.add_argument("--nude-threshold", type=float, default=0.55)
    parser.add_argument("--sexy-threshold", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=None, help="Gioi han so anh de test")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="File log debug. Mac dinh: <output>\\debug.log",
    )
    parser.add_argument(
        "--debug-log",
        action="store_true",
        help="Ghi log chi tiet tung anh. Cham hon, chi nen bat khi debug.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="So anh moi cap nhat progress mot lan trong GUI/callback.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    result = scan_and_classify(
        root=args.folder,
        output_dir=args.output,
        mode=args.mode,
        batch_size=args.batch_size,
        nude_threshold=args.nude_threshold,
        sexy_threshold=args.sexy_threshold,
        limit=args.limit,
        log_path=args.log,
        debug_log=args.debug_log,
        progress_interval=args.progress_interval,
        device=args.device,
    )
    print(
        "Done. "
        f"seen={result.total_seen}, processed={result.processed}, "
        f"skipped={result.skipped}, errors={result.errors}, "
        f"log={result.log_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
