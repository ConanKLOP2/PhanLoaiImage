from __future__ import annotations

import argparse
import csv
import inspect
import logging
import os
import shutil
import sys
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

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
TEMP_DIR_NAME = "_tmp_ascii_paths"


@dataclass(frozen=True)
class ScanResult:
    total_seen: int
    processed: int
    skipped: int
    errors: int
    batch_errors: int
    log_path: Path
    providers: list[str]


@dataclass(frozen=True)
class TransferJob:
    source: Path
    destination: Path
    category: str
    status: str
    reason: str
    detection_error: bool = False


@dataclass(frozen=True)
class TransferOutcome:
    job: TransferJob
    ok: bool
    error: str = ""


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


def force_onnxruntime_providers(providers: list[str]) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        return

    if "CUDAExecutionProvider" in providers:
        try:
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        except Exception:
            pass
        try:
            import torch  # noqa: F401
        except Exception:
            pass

    if getattr(ort.InferenceSession, "_phanloai_wrapped", False):
        return

    original_session = ort.InferenceSession

    def patched_session(*args, **kwargs):
        if "providers" not in kwargs:
            kwargs["providers"] = providers
        return original_session(*args, **kwargs)

    patched_session._phanloai_wrapped = True  # type: ignore[attr-defined]
    ort.InferenceSession = patched_session


def load_detector(
    device: str = "auto",
    engine: str = "onnx",
    preprocess_workers: int = 4,
):
    providers = get_onnx_providers(device)
    force_onnxruntime_providers(providers)

    if engine == "onnx":
        from fast_onnx_detector import FastOnnxNudeDetector

        detector = FastOnnxNudeDetector(
            providers=providers,
            preprocess_workers=preprocess_workers,
        )
        if device == "gpu" and "CUDAExecutionProvider" not in detector.providers:
            raise RuntimeError(
                "device=gpu was requested, but the ONNX session did not activate CUDAExecutionProvider."
            )
        return detector, detector.providers

    try:
        from nudenet import NudeDetector
    except ImportError as exc:
        raise RuntimeError(
            "Chua cai nudenet. Hay chay: pip install -r requirements.txt"
        ) from exc

    signature = inspect.signature(NudeDetector)
    kwargs = {}
    if "providers" in signature.parameters:
        kwargs["providers"] = providers
    elif "provider" in signature.parameters:
        kwargs["provider"] = providers[0]

    try:
        detector = NudeDetector(**kwargs)
    except TypeError:
        detector = NudeDetector()

    actual_providers = providers
    onnx_session = getattr(detector, "onnx_session", None)
    if onnx_session is not None and hasattr(onnx_session, "get_providers"):
        actual_providers = onnx_session.get_providers()
    if device == "gpu" and "CUDAExecutionProvider" not in actual_providers:
        raise RuntimeError(
            "device=gpu was requested, but the ONNX session did not activate CUDAExecutionProvider."
        )
    return detector, actual_providers


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
    temp_dir = (output_dir / TEMP_DIR_NAME).resolve()
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname != OUTPUT_DIR_NAME
            and (current_path / dirname).resolve() not in {output_dir, temp_dir}
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
            if source and status in {"moved", "copied", "copyd", "skipped_missing"}:
                done.add(source)
    return done


def read_done_manifests(manifest_paths: Iterable[Path]) -> set[str]:
    done: set[str] = set()
    for manifest_path in manifest_paths:
        done.update(read_done_manifest(manifest_path))
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


def reserve_destination(
    destination_dir: Path, source: Path, reserved_destinations: set[Path]
) -> Path:
    destination = destination_dir / source.name
    if not destination.exists() and destination not in reserved_destinations:
        reserved_destinations.add(destination)
        return destination

    stem = source.stem
    suffix = source.suffix
    index = 1
    while True:
        candidate = destination_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists() and candidate not in reserved_destinations:
            reserved_destinations.add(candidate)
            return candidate
        index += 1


def transfer_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        shutil.move(str(source), str(destination))


def transfer_status(mode: str) -> str:
    return "copied" if mode == "copy" else "moved"


def run_transfer(job: TransferJob, mode: str) -> TransferOutcome:
    try:
        transfer_file(job.source, job.destination, mode)
        return TransferOutcome(job=job, ok=True)
    except Exception as exc:
        return TransferOutcome(job=job, ok=False, error=repr(exc))


def needs_ascii_staging(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def stage_ascii_file(source: Path, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    staged = temp_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        os.link(source, staged)
    except OSError:
        shutil.copy2(source, staged)
    return staged


@contextmanager
def staged_detector_paths(paths: list[Path], temp_dir: Path) -> Iterator[list[str]]:
    staged_paths: list[Path] = []
    detector_paths: list[str] = []
    try:
        for path in paths:
            if needs_ascii_staging(path):
                staged = stage_ascii_file(path, temp_dir)
                staged_paths.append(staged)
                detector_paths.append(str(staged))
            else:
                detector_paths.append(str(path))
        yield detector_paths
    finally:
        for staged in staged_paths:
            try:
                staged.unlink()
            except OSError:
                pass


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
    mode: str = "copy",
    batch_size: int = 128,
    nude_threshold: float = 0.55,
    sexy_threshold: float = 0.55,
    limit: int | None = None,
    progress: Callable[[int, int, Path, str], None] | None = None,
    log_path: Path | None = None,
    debug_log: bool = False,
    progress_interval: int = 25,
    device: str = "auto",
    transfer_workers: int = 0,
    engine: str = "onnx",
    preprocess_workers: int = 4,
    output_strategy: str = "root",
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
    if engine not in {"onnx", "nudenet"}:
        raise ValueError("engine phai la 'onnx' hoac 'nudenet'")
    if output_strategy not in {"root", "per-folder"}:
        raise ValueError("output_strategy phai la 'root' hoac 'per-folder'")
    if output_dir is not None and output_strategy == "per-folder":
        raise ValueError("output_dir chi ho tro voi output_strategy='root'")
    if progress_interval < 1:
        progress_interval = 1
    if transfer_workers < 0:
        raise ValueError("transfer_workers phai >= 0")
    if preprocess_workers < 1:
        raise ValueError("preprocess_workers phai >= 1")

    output_dir = (output_dir or root / OUTPUT_DIR_NAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    log_path = (log_path or output_dir / "debug.log").resolve()
    temp_dir = output_dir / TEMP_DIR_NAME
    if not debug_log and log_path.exists():
        try:
            log_path.unlink()
        except OSError:
            pass
    logger = setup_logger(log_path, debug=debug_log)

    def output_for_source(path: Path) -> Path:
        if output_strategy == "per-folder":
            return (path.parent / OUTPUT_DIR_NAME).resolve()
        return output_dir

    for category in ("nude", "sexy", "normal", "errors"):
        (output_dir / category).mkdir(parents=True, exist_ok=True)

    detector = None
    detector, providers = load_detector(device, engine, preprocess_workers)
    has_detect_batch = hasattr(detector, "detect_batch")
    use_ascii_staging = not getattr(detector, "accepts_unicode_paths", False)
    if debug_log:
        logger.debug(
            "Started root=%s output_dir=%s mode=%s batch_size=%s limit=%s device=%s engine=%s providers=%s has_detect_batch=%s preprocess_workers=%s",
            root,
            output_dir,
            mode,
            batch_size,
            limit,
            device,
            engine,
            providers,
            has_detect_batch,
            preprocess_workers,
        )
    all_images: list[Path] = []
    for path in iter_images(root, output_dir):
        if limit is not None and len(all_images) >= limit:
            break
        all_images.append(path)

    if output_strategy == "per-folder":
        manifest_paths = {
            output_for_source(path) / MANIFEST_NAME for path in all_images
        }
        done_sources = read_done_manifests(manifest_paths)
    else:
        done_sources = read_done_manifest(manifest_path)

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
    batch_errors = 0
    missing_batch_logged = False
    completed = 0
    reserved_destinations: set[Path] = set()
    manifest_writers: dict[Path, ManifestWriter] = {}
    worker_count = transfer_workers or (2 if mode == "copy" else 1)
    max_pending_transfers = max(worker_count * 8, batch_size * 2)
    pending_transfers: set[Future[TransferOutcome]] = set()

    def maybe_report(done: int, total: int, path: Path, category: str) -> None:
        if not progress:
            return
        if done == total or done % progress_interval == 0 or category == "errors":
            progress(done, total, path, category)

    def handle_transfer_outcome(
        outcome: TransferOutcome,
        bar,
    ) -> None:
        nonlocal processed, errors, completed
        job = outcome.job
        target_output_dir = job.destination.parent.parent
        manifest = manifest_writers[target_output_dir]
        if outcome.ok:
            manifest.append(
                job.source,
                job.destination,
                job.category,
                job.status,
                job.reason,
            )
            if job.detection_error:
                errors += 1
            else:
                processed += 1
        else:
            logger.error(
                "Transfer failed: file=%s category=%s error=%s",
                job.source,
                job.category,
                outcome.error,
            )
            manifest.append(
                job.source,
                None,
                job.category,
                "error",
                outcome.error,
            )
            errors += 1
        completed += 1
        bar.update(1)
        maybe_report(completed, total_pending, job.source, job.category)

    def drain_transfers(
        bar,
        block: bool = False,
    ) -> None:
        if not pending_transfers:
            return
        if block:
            done, _ = wait(pending_transfers, return_when=FIRST_COMPLETED)
        else:
            done = {future for future in pending_transfers if future.done()}
        for future in done:
            pending_transfers.remove(future)
            handle_transfer_outcome(future.result(), bar)

    def submit_transfer(
        executor: ThreadPoolExecutor,
        bar,
        source: Path,
        category: str,
        reason: str,
        status: str,
        detection_error: bool = False,
    ) -> None:
        target_output_dir = output_for_source(source)
        for folder_name in ("nude", "sexy", "normal", "errors"):
            (target_output_dir / folder_name).mkdir(parents=True, exist_ok=True)
        if target_output_dir not in manifest_writers:
            writer = ManifestWriter(target_output_dir / MANIFEST_NAME)
            manifest_writers[target_output_dir] = writer.__enter__()
        destination = reserve_destination(
            target_output_dir / category, source, reserved_destinations
        )
        job = TransferJob(
            source=source,
            destination=destination,
            category=category,
            status=status,
            reason=reason,
            detection_error=detection_error,
        )
        pending_transfers.add(executor.submit(run_transfer, job, mode))
        while len(pending_transfers) >= max_pending_transfers:
            drain_transfers(bar, block=True)

    try:
        if output_strategy == "root":
            root_writer = ManifestWriter(manifest_path)
            manifest_writers[output_dir] = root_writer.__enter__()

        with ThreadPoolExecutor(max_workers=worker_count) as transfer_executor, tqdm(
            total=total_pending, unit="img", desc="Classifying"
        ) as bar:
            for batch in chunked(pending_paths, batch_size):
                drain_transfers(bar, block=False)
                existing_batch = []
                for path in batch:
                    if path.exists():
                        existing_batch.append(path)
                    else:
                        missing_output_dir = output_for_source(path)
                        if missing_output_dir not in manifest_writers:
                            writer = ManifestWriter(missing_output_dir / MANIFEST_NAME)
                            manifest_writers[missing_output_dir] = writer.__enter__()
                        manifest_writers[missing_output_dir].append(
                            path, None, "", "skipped_missing", "missing"
                        )
                        skipped += 1
                        completed += 1
                        bar.update(1)
                        maybe_report(completed, total_pending, path, "skipped")

                if not existing_batch:
                    continue

                batch_error: Exception | None = None
                if has_detect_batch:
                    try:
                        if use_ascii_staging:
                            with staged_detector_paths(
                                existing_batch, temp_dir
                            ) as detector_paths:
                                predictions = detector.detect_batch(detector_paths)
                        else:
                            predictions = detector.detect_batch(existing_batch)
                        if len(predictions) != len(existing_batch):
                            raise RuntimeError(
                                "detect_batch returned "
                                f"{len(predictions)} results for {len(existing_batch)} images"
                            )
                    except Exception as exc:
                        batch_error = exc
                        batch_errors += 1
                        logger.exception(
                            "Batch failed. Falling back to single-file detection. batch_size=%s first_file=%s error=%r",
                            len(existing_batch),
                            existing_batch[0] if existing_batch else "",
                            exc,
                        )
                        predictions = None
                else:
                    if not missing_batch_logged:
                        batch_errors += 1
                        logger.error(
                            "detect_batch is not available on NudeDetector. Using single-file detection. Upgrade nudenet to >=3.4.2 for real batch inference."
                        )
                        missing_batch_logged = True
                    predictions = None

                if predictions is None:
                    if debug_log and batch_error is None:
                        logger.debug(
                            "detect_batch not available. Using single-file detection. batch_size=%s first_file=%s",
                            len(existing_batch),
                            existing_batch[0] if existing_batch else "",
                        )
                    predictions = []
                    for path in existing_batch:
                        try:
                            if use_ascii_staging:
                                with staged_detector_paths([path], temp_dir) as detector_paths:
                                    predictions.append(detector.detect(detector_paths[0]))
                            else:
                                predictions.append(detector.detect(path))
                        except Exception as single_exc:
                            logger.exception(
                                "Single-file detection failed: file=%s error=%r",
                                path,
                                single_exc,
                            )
                            error_reason = (
                                f"{type(single_exc).__name__}: {single_exc}\n"
                                f"{traceback.format_exc()}"
                            )
                            submit_transfer(
                                transfer_executor,
                                bar,
                                path,
                                "errors",
                                error_reason,
                                "error",
                                detection_error=True,
                            )
                            continue

                        category, reason = classify_detection(
                            predictions[-1], nude_threshold, sexy_threshold
                        )
                        submit_transfer(
                            transfer_executor,
                            bar,
                            path,
                            category,
                            reason,
                            transfer_status(mode),
                        )
                    continue

                for path, detections in zip(existing_batch, predictions):
                    category, reason = classify_detection(
                        detections, nude_threshold, sexy_threshold
                    )
                    submit_transfer(
                        transfer_executor,
                        bar,
                        path,
                    category,
                    reason,
                    transfer_status(mode),
                )

            while pending_transfers:
                drain_transfers(bar, block=True)
    finally:
        for writer in manifest_writers.values():
            writer.__exit__(None, None, None)
        detector_close = getattr(detector, "close", None)
        if callable(detector_close):
            detector_close()

    if debug_log:
        logger.debug(
            "Finished seen=%s processed=%s skipped=%s errors=%s batch_errors=%s log=%s",
            len(all_images),
            processed,
            skipped,
            errors,
            batch_errors,
            log_path,
        )
    if not debug_log and errors == 0 and log_path.exists():
        try:
            if log_path.stat().st_size == 0:
                log_path.unlink()
        except OSError:
            pass
    return ScanResult(
        total_seen=len(all_images),
        processed=processed,
        skipped=skipped,
        errors=errors,
        batch_errors=batch_errors,
        log_path=log_path,
        providers=providers,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phan loai anh thanh nude, sexy, normal bang NudeNet."
    )
    parser.add_argument(
        "folders",
        type=Path,
        nargs="+",
        help="Mot hoac nhieu thu muc chua anh can phan loai",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Thu muc output. Mac dinh: <folder>\\_classified",
    )
    parser.add_argument(
        "--mode",
        choices=["move", "copy"],
        default="copy",
        help="copy de giu file goc, move de chuyen file",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="auto dung GPU neu onnxruntime CUDA kha dung, gpu bat buoc CUDA, cpu bat buoc CPU.",
    )
    parser.add_argument(
        "--engine",
        choices=["onnx", "nudenet"],
        default="onnx",
        help="onnx chay truc tiep nhanh hon, nudenet dung wrapper thu vien.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=4,
        help="So worker CPU doc/decode/preprocess anh cho engine onnx.",
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
    parser.add_argument(
        "--transfer-workers",
        type=int,
        default=0,
        help="So worker copy/move nen. 0 = tu dong: copy dung 2, move dung 1.",
    )
    parser.add_argument(
        "--output-strategy",
        choices=["root", "per-folder"],
        default="root",
        help="root gom ket qua vao folder goc; per-folder tao _classified rieng trong tung folder co anh.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.output and len(args.folders) > 1:
        raise SystemExit("--output chi dung voi mot folder. Multi-folder dung output mac dinh rieng tung folder.")
    if args.output and args.output_strategy == "per-folder":
        raise SystemExit("--output khong dung chung voi --output-strategy per-folder.")

    for folder in args.folders:
        result = scan_and_classify(
            root=folder,
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
            transfer_workers=args.transfer_workers,
            engine=args.engine,
            preprocess_workers=args.preprocess_workers,
            output_strategy=args.output_strategy,
        )
        message = (
            f"Done: {folder}. "
            f"seen={result.total_seen}, processed={result.processed}, "
            f"skipped={result.skipped}, errors={result.errors}, "
            f"batch_errors={result.batch_errors}, "
            f"providers={result.providers}"
        )
        if result.errors or result.batch_errors:
            message += f", log={result.log_path}"
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
