from __future__ import annotations

import argparse
import logging
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from manifest import (
    MANIFEST_NAME,
    ManifestWriter,
    read_done_manifest,
    read_done_manifests,
)
from policy import NUDE_LABELS, SEXY_LABELS, classify_detection
from scanner import IMAGE_EXTENSIONS, OUTPUT_DIR_NAME, is_image, iter_images
from transfer import (
    DestinationAllocator,
    TransferJob,
    TransferOutcome,
    run_transfer,
    transfer_file,
    transfer_status,
)

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


CATEGORIES = ("nude", "sexy", "normal", "errors")

__all__ = [
    "CATEGORIES",
    "IMAGE_EXTENSIONS",
    "MANIFEST_NAME",
    "NUDE_LABELS",
    "OUTPUT_DIR_NAME",
    "SEXY_LABELS",
    "ScanResult",
    "TransferJob",
    "TransferOutcome",
    "classify_detection",
    "is_image",
    "iter_images",
    "load_detector",
    "main",
    "read_done_manifest",
    "read_done_manifests",
    "run_transfer",
    "scan_and_classify",
    "scan_folders",
    "transfer_file",
    "transfer_status",
]


@dataclass(frozen=True)
class ScanResult:
    total_seen: int
    processed: int
    skipped: int
    errors: int
    batch_errors: int
    log_path: Path
    providers: list[str]
    cancelled: bool


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


def load_detector(
    device: str = "auto",
    preprocess_workers: int = 4,
    fast_decode: bool = False,
):
    from fast_onnx_detector import FastOnnxNudeDetector

    providers = get_onnx_providers(device)
    detector = FastOnnxNudeDetector(
        providers=providers,
        preprocess_workers=preprocess_workers,
        fast_decode=fast_decode,
    )
    if device == "gpu" and "CUDAExecutionProvider" not in detector.providers:
        detector.close()
        raise RuntimeError(
            "device=gpu was requested, but the ONNX session did not activate CUDAExecutionProvider."
        )
    return detector, detector.providers


class _LazyDirFileHandler(logging.FileHandler):
    """FileHandler that creates its parent directory only when it first writes."""

    def _open(self):
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return super()._open()


def setup_logger(log_path: Path, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("phan_loai_image")
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = _LazyDirFileHandler(log_path, encoding="utf-8", delay=True)
    file_handler.setLevel(logging.DEBUG if debug else logging.ERROR)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def chunked(items: Iterable[Path], size: int) -> Iterator[list[Path]]:
    batch: list[Path] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class _FolderRun:
    """State and steps for classifying the pending images of one folder."""

    def __init__(
        self,
        *,
        detector,
        output_dir: Path,
        output_strategy: str,
        mode: str,
        batch_size: int,
        nude_threshold: float,
        sexy_threshold: float,
        transfer_workers: int,
        progress: Callable[[int, int, Path, str], None] | None,
        progress_interval: int,
        logger: logging.Logger,
        debug_log: bool,
        cancel_event,
    ) -> None:
        self.detector = detector
        self.output_dir = output_dir
        self.output_strategy = output_strategy
        self.mode = mode
        self.batch_size = batch_size
        self.nude_threshold = nude_threshold
        self.sexy_threshold = sexy_threshold
        self.progress = progress
        self.progress_interval = progress_interval
        self.logger = logger
        self.debug_log = debug_log
        self.cancel_event = cancel_event

        self.worker_count = transfer_workers or (2 if mode == "copy" else 1)
        self.max_pending_transfers = max(self.worker_count * 8, batch_size * 2)
        self.pending_transfers: set[Future[TransferOutcome]] = set()
        self.manifest_writers: dict[Path, ManifestWriter] = {}
        self.prepared_dirs: set[Path] = set()
        self.allocator = DestinationAllocator()

        self.total_pending = 0
        self.completed = 0
        self.processed = 0
        self.skipped = 0
        self.errors = 0
        self.batch_errors = 0
        self.cancelled = False
        self.bar = None
        self.executor: ThreadPoolExecutor | None = None

    # -- output bookkeeping -------------------------------------------------

    def output_for_source(self, path: Path) -> Path:
        if self.output_strategy == "per-folder":
            return (path.parent / OUTPUT_DIR_NAME).resolve()
        return self.output_dir

    def prepare_output(self, target_output_dir: Path) -> ManifestWriter:
        if target_output_dir not in self.prepared_dirs:
            for folder_name in CATEGORIES:
                (target_output_dir / folder_name).mkdir(parents=True, exist_ok=True)
            self.manifest_writers[target_output_dir] = ManifestWriter(
                target_output_dir / MANIFEST_NAME
            ).__enter__()
            self.prepared_dirs.add(target_output_dir)
        return self.manifest_writers[target_output_dir]

    def close_manifests(self) -> None:
        for writer in self.manifest_writers.values():
            writer.__exit__(None, None, None)

    # -- progress -----------------------------------------------------------

    def is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def advance(self, path: Path, category: str) -> None:
        self.completed += 1
        self.bar.update(1)
        if not self.progress:
            return
        if (
            self.completed == self.total_pending
            or self.completed % self.progress_interval == 0
            or category == "errors"
        ):
            self.progress(self.completed, self.total_pending, path, category)

    # -- transfers ----------------------------------------------------------

    def handle_transfer_outcome(self, outcome: TransferOutcome) -> None:
        job = outcome.job
        manifest = self.manifest_writers[job.destination.parent.parent]
        if outcome.ok:
            manifest.append(
                job.source, job.destination, job.category, job.status, job.reason
            )
            if job.detection_error:
                self.errors += 1
            else:
                self.processed += 1
        else:
            self.logger.error(
                "Transfer failed: file=%s category=%s error=%s",
                job.source,
                job.category,
                outcome.error,
            )
            manifest.append(job.source, None, job.category, "error", outcome.error)
            self.errors += 1
        self.advance(job.source, job.category)

    def drain_transfers(self, block: bool = False) -> None:
        if not self.pending_transfers:
            return
        if block:
            done, _ = wait(self.pending_transfers, return_when=FIRST_COMPLETED)
        else:
            done = {future for future in self.pending_transfers if future.done()}
        for future in done:
            self.pending_transfers.remove(future)
            self.handle_transfer_outcome(future.result())

    def submit_transfer(
        self,
        source: Path,
        category: str,
        reason: str,
        status: str,
        detection_error: bool = False,
    ) -> None:
        target_output_dir = self.output_for_source(source)
        self.prepare_output(target_output_dir)
        job = TransferJob(
            source=source,
            destination=self.allocator.reserve(target_output_dir / category, source),
            category=category,
            status=status,
            reason=reason,
            detection_error=detection_error,
        )
        self.pending_transfers.add(self.executor.submit(run_transfer, job, self.mode))
        while len(self.pending_transfers) >= self.max_pending_transfers:
            self.drain_transfers(block=True)

    # -- per-image results --------------------------------------------------

    def record_missing(self, path: Path) -> None:
        self.prepare_output(self.output_for_source(path)).append(
            path, None, "", "skipped_missing", "missing"
        )
        self.skipped += 1
        self.advance(path, "skipped")

    def record_detection(self, path: Path, detections: list[dict]) -> None:
        category, reason = classify_detection(
            detections, self.nude_threshold, self.sexy_threshold
        )
        self.submit_transfer(path, category, reason, transfer_status(self.mode))

    def record_detection_error(self, path: Path, exc: BaseException) -> None:
        if isinstance(exc, FileNotFoundError):
            self.record_missing(path)
            return
        self.logger.error(
            "Detection failed: file=%s error=%r\n%s", path, exc, format_exception(exc)
        )
        self.submit_transfer(
            path,
            "errors",
            f"{type(exc).__name__}: {exc}\n{format_exception(exc)}",
            "error",
            detection_error=True,
        )

    # -- detection ----------------------------------------------------------

    def detect_one_by_one(self, paths: list[Path]) -> None:
        for path in paths:
            if self.is_cancelled():
                self.cancelled = True
                return
            try:
                detections = self.detector.detect(path)
            except Exception as exc:
                self.record_detection_error(path, exc)
                continue
            self.record_detection(path, detections)

    def record_results(self, paths: list[Path], results: list) -> None:
        for path, result in zip(paths, results):
            if self.is_cancelled():
                self.cancelled = True
                return
            if isinstance(result, Exception):
                self.record_detection_error(path, result)
            else:
                self.record_detection(path, result)

    def fall_back_to_singles(self, batch: list[Path], exc: Exception) -> None:
        self.batch_errors += 1
        self.logger.error(
            "Batch failed. Falling back to single-file detection. batch_size=%s first_file=%s error=%r",
            len(batch),
            batch[0],
            exc,
        )
        self.detect_one_by_one(batch)

    def process_batch(self, batch: list[Path]) -> None:
        try:
            results = self.detector.detect_batch(batch)
            if len(results) != len(batch):
                raise RuntimeError(
                    f"detect_batch returned {len(results)} results for {len(batch)} images"
                )
        except Exception as exc:
            self.fall_back_to_singles(batch, exc)
            return
        self.record_results(batch, results)

    def run_pipelined(self, batches: Iterator[list[Path]]) -> None:
        """Preprocess batch N+1 on the CPU pool while batch N is on the GPU."""
        current = next(batches, None)
        prepared = self.detector.prepare_batch(current) if current else None
        while current:
            if self.is_cancelled():
                self.cancelled = True
                return
            self.drain_transfers(block=False)
            following = next(batches, None)
            following_prepared = (
                self.detector.prepare_batch(following) if following else None
            )
            try:
                results = self.detector.run_prepared(prepared)
            except Exception as exc:
                self.fall_back_to_singles(current, exc)
            else:
                self.record_results(current, results)
            if self.cancelled:
                return
            current, prepared = following, following_prepared

    def run(self, pending_paths: list[Path]) -> None:
        self.total_pending = len(pending_paths)
        batches = chunked(pending_paths, self.batch_size)
        pipelined = hasattr(self.detector, "prepare_batch") and hasattr(
            self.detector, "run_prepared"
        )
        try:
            with ThreadPoolExecutor(max_workers=self.worker_count) as executor, tqdm(
                total=self.total_pending, unit="img", desc="Classifying"
            ) as bar:
                self.executor = executor
                self.bar = bar
                if pipelined:
                    self.run_pipelined(batches)
                else:
                    for batch in batches:
                        if self.is_cancelled():
                            self.cancelled = True
                            break
                        self.drain_transfers(block=False)
                        self.process_batch(batch)
                        if self.cancelled:
                            break
                while self.pending_transfers:
                    self.drain_transfers(block=True)
        finally:
            self.close_manifests()


def validate_options(
    root: Path,
    mode: str,
    batch_size: int,
    device: str,
    output_strategy: str,
    output_dir: Path | None,
    transfer_workers: int,
    preprocess_workers: int,
) -> None:
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Thu muc khong ton tai: {root}")
    if mode not in {"move", "copy"}:
        raise ValueError("mode phai la 'move' hoac 'copy'")
    if batch_size < 1:
        raise ValueError("batch_size phai >= 1")
    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device phai la 'auto', 'cpu', hoac 'gpu'")
    if output_strategy not in {"root", "per-folder"}:
        raise ValueError("output_strategy phai la 'root' hoac 'per-folder'")
    if output_dir is not None and output_strategy == "per-folder":
        raise ValueError("output_dir chi ho tro voi output_strategy='root'")
    if transfer_workers < 0:
        raise ValueError("transfer_workers phai >= 0")
    if preprocess_workers < 1:
        raise ValueError("preprocess_workers phai >= 1")


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
    preprocess_workers: int = 4,
    output_strategy: str = "root",
    cancel_event=None,
    fast_decode: bool = False,
    detector=None,
    providers: list[str] | None = None,
) -> ScanResult:
    """Classify one folder.

    Pass an already loaded `detector` (and its `providers`) to reuse a model
    across folders; the caller then owns closing it.
    """
    root = root.resolve()
    validate_options(
        root,
        mode,
        batch_size,
        device,
        output_strategy,
        output_dir,
        transfer_workers,
        preprocess_workers,
    )
    progress_interval = max(1, progress_interval)

    output_dir = (output_dir or root / OUTPUT_DIR_NAME).resolve()
    manifest_path = output_dir / MANIFEST_NAME
    log_path = (log_path or output_dir / "debug.log").resolve()
    if not debug_log and log_path.exists():
        try:
            log_path.unlink()
        except OSError:
            pass
    logger = setup_logger(log_path, debug=debug_log)

    if output_strategy == "root":
        for category in CATEGORIES:
            (output_dir / category).mkdir(parents=True, exist_ok=True)

    owns_detector = detector is None
    if owns_detector:
        detector, providers = load_detector(device, preprocess_workers, fast_decode)
    providers = list(providers or [])

    try:
        if debug_log:
            logger.debug(
                "Started root=%s output_dir=%s mode=%s batch_size=%s limit=%s device=%s providers=%s preprocess_workers=%s fast_decode=%s",
                root,
                output_dir,
                mode,
                batch_size,
                limit,
                device,
                providers,
                preprocess_workers,
                fast_decode,
            )
        all_images: list[Path] = []
        for path in iter_images(root, output_dir):
            if limit is not None and len(all_images) >= limit:
                break
            all_images.append(path)

        run = _FolderRun(
            detector=detector,
            output_dir=output_dir,
            output_strategy=output_strategy,
            mode=mode,
            batch_size=batch_size,
            nude_threshold=nude_threshold,
            sexy_threshold=sexy_threshold,
            transfer_workers=transfer_workers,
            progress=progress,
            progress_interval=progress_interval,
            logger=logger,
            debug_log=debug_log,
            cancel_event=cancel_event,
        )

        if output_strategy == "per-folder":
            done_sources = read_done_manifests(
                {run.output_for_source(path) / MANIFEST_NAME for path in all_images}
            )
        else:
            done_sources = read_done_manifest(manifest_path)
        pending_paths = [path for path in all_images if str(path) not in done_sources]
        if debug_log:
            logger.debug(
                "images_found=%s pending=%s already_done=%s",
                len(all_images),
                len(pending_paths),
                len(all_images) - len(pending_paths),
            )

        run.run(pending_paths)
    finally:
        if owns_detector:
            detector_close = getattr(detector, "close", None)
            if callable(detector_close):
                detector_close()

    if debug_log:
        logger.debug(
            "Finished seen=%s processed=%s skipped=%s errors=%s batch_errors=%s log=%s",
            len(all_images),
            run.processed,
            run.skipped,
            run.errors,
            run.batch_errors,
            log_path,
        )
    if not debug_log and run.errors == 0 and log_path.exists():
        try:
            if log_path.stat().st_size == 0:
                log_path.unlink()
        except OSError:
            pass
    return ScanResult(
        total_seen=len(all_images),
        processed=run.processed,
        skipped=run.skipped,
        errors=run.errors,
        batch_errors=run.batch_errors,
        log_path=log_path,
        providers=providers,
        cancelled=run.cancelled,
    )


def scan_folders(
    roots: list[Path],
    *,
    device: str = "auto",
    preprocess_workers: int = 4,
    fast_decode: bool = False,
    cancel_event=None,
    on_folder: Callable[[int, int, Path], None] | None = None,
    **options,
) -> list[tuple[Path, ScanResult]]:
    """Classify several folders, loading the model once for all of them."""
    if not roots:
        return []
    for root in roots:
        validate_options(
            Path(root).resolve(),
            options.get("mode", "copy"),
            options.get("batch_size", 128),
            device,
            options.get("output_strategy", "root"),
            options.get("output_dir"),
            options.get("transfer_workers", 0),
            preprocess_workers,
        )

    detector, providers = load_detector(device, preprocess_workers, fast_decode)
    results: list[tuple[Path, ScanResult]] = []
    try:
        for index, root in enumerate(roots, start=1):
            if cancel_event is not None and cancel_event.is_set():
                break
            if on_folder:
                on_folder(index, len(roots), root)
            result = scan_and_classify(
                root=root,
                device=device,
                preprocess_workers=preprocess_workers,
                cancel_event=cancel_event,
                detector=detector,
                providers=providers,
                **options,
            )
            results.append((root, result))
    finally:
        detector.close()
    return results


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
        choices=["onnx"],
        default="onnx",
        help=argparse.SUPPRESS,  # kept so old command lines keep working
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=4,
        help="So worker CPU doc/decode/preprocess anh.",
    )
    parser.add_argument(
        "--fast-decode",
        action="store_true",
        help="Decode JPEG o do phan giai giam (nhanh hon voi anh lon, score co the lech nhe).",
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
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.output and len(args.folders) > 1:
        raise SystemExit("--output chi dung voi mot folder. Multi-folder dung output mac dinh rieng tung folder.")
    if args.output and args.output_strategy == "per-folder":
        raise SystemExit("--output khong dung chung voi --output-strategy per-folder.")

    results = scan_folders(
        args.folders,
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
        preprocess_workers=args.preprocess_workers,
        output_strategy=args.output_strategy,
        fast_decode=args.fast_decode,
    )
    for folder, result in results:
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
