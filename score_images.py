"""Run the model over folders and save raw detections to scores.jsonl.

Nothing is copied or moved. The file lets accuracy tooling re-classify with any
threshold offline:

    python score_images.py "D:\\Images" --out scores.jsonl --device gpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from classify_images import chunked, load_detector
from scanner import OUTPUT_DIR_NAME, iter_images


def read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8") as file:
            for line in file:
                try:
                    done.add(json.loads(line)["source"])
                except (json.JSONDecodeError, KeyError):
                    continue  # a half-written last line after a crash
    return done


def detections_to_json(detections: list[dict]) -> list[dict]:
    return [
        {"class": d["class"], "score": round(float(d["score"]), 5), "box": d.get("box")}
        for d in detections
    ]


def score_folders(
    roots: list[Path],
    out_path: Path,
    detector,
    batch_size: int = 64,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel_event=None,
) -> dict[str, int]:
    """Append one JSON line per image. Images already in out_path are skipped."""
    out_path = Path(out_path)
    done = read_done(out_path)
    pending: list[Path] = []
    for root in roots:
        root = Path(root).resolve()
        for path in iter_images(root, root / OUTPUT_DIR_NAME):
            if limit is not None and len(pending) >= limit:
                break
            if str(path) not in done:
                pending.append(path)

    counts = {"scored": 0, "errors": 0, "skipped": len(done)}
    pipelined = hasattr(detector, "prepare_batch") and hasattr(detector, "run_prepared")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as out:
        batches = chunked(pending, batch_size)
        current = next(batches, None)
        prepared = detector.prepare_batch(current) if current and pipelined else None
        while current:
            if cancel_event is not None and cancel_event.is_set():
                break
            following = next(batches, None)
            following_prepared = (
                detector.prepare_batch(following) if following and pipelined else None
            )
            try:
                results = (
                    detector.run_prepared(prepared)
                    if pipelined
                    else detector.detect_batch(current)
                )
            except Exception:
                results = []
                for path in current:
                    try:
                        results.append(detector.detect(path))
                    except Exception as exc:
                        results.append(exc)

            for path, result in zip(current, results):
                if isinstance(result, Exception):
                    row = {"source": str(path), "detections": [], "error": repr(result)}
                    counts["errors"] += 1
                else:
                    row = {
                        "source": str(path),
                        "detections": detections_to_json(result),
                        "error": None,
                    }
                    counts["scored"] += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if progress:
                progress(counts["scored"] + counts["errors"], len(pending))
            current, prepared = following, following_prepared
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("scores.jsonl"))
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preprocess-workers", type=int, default=6)
    parser.add_argument("--fast-decode", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    for folder in args.folders:
        if not folder.is_dir():
            raise SystemExit(f"Thu muc khong ton tai: {folder}")

    detector, providers = load_detector(args.device, args.preprocess_workers, args.fast_decode)
    try:
        counts = score_folders(
            args.folders,
            args.out,
            detector,
            batch_size=args.batch_size,
            limit=args.limit,
            progress=lambda done, total: print(f"\r{done}/{total}", end="", flush=True),
        )
    finally:
        detector.close()
    print(
        f"\nscored={counts['scored']} errors={counts['errors']} "
        f"already_done={counts['skipped']} providers={providers} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
