"""Benchmark and golden-check the detector on a folder of images.

    python bench.py --dir D:\\Images --limit 2000 --device gpu
    python bench.py --synthetic 300                      # no images needed
    python bench.py --dir D:\\Images --save-golden golden.json
    python bench.py --dir D:\\Images --compare-golden golden.json

Reports images/second for each stage so you can see which one limits a run,
and (with --compare-golden) checks that categories did not change.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from classify_images import get_onnx_providers
from fast_onnx_detector import FastOnnxNudeDetector
from policy import classify_detection
from scanner import iter_images


def make_synthetic(directory: Path, count: int) -> list[Path]:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sizes = [(640, 480), (1280, 720), (1920, 1080), (3000, 2000)]
    paths = []
    for index in range(count):
        width, height = sizes[index % len(sizes)]
        base = rng.integers(0, 256, size=(height // 8, width // 8, 3), dtype=np.uint8)
        image = cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR)
        path = directory / f"synthetic_{index:05d}.jpg"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def timed(function, *args):
    start = time.perf_counter()
    value = function(*args)
    return value, time.perf_counter() - start


def rate(count: int, seconds: float) -> str:
    return f"{count / seconds:8.1f} img/s ({seconds:6.2f}s)" if seconds else "      n/a"


def run_bench(detector: FastOnnxNudeDetector, paths: list[Path], batch_size: int) -> dict:
    stages = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    detections: dict[str, list] = {}

    total_start = time.perf_counter()
    for batch in chunks(paths, batch_size):
        prepared, seconds = timed(
            lambda b: [f.result() if not f.exception() else None
                       for f in detector.prepare_batch(b).futures],
            batch,
        )
        stages["preprocess"] += seconds

        good = [(p, item) for p, item in zip(batch, prepared) if item is not None]
        if not good:
            continue
        batch_input = np.vstack([item[0] for _, item in good])
        outputs, seconds = timed(
            lambda x: detector.session.run(None, {detector.input_name: x})[0], batch_input
        )
        stages["inference"] += seconds

        def post():
            return [
                detector._postprocess(outputs[i], item[1])
                for i, (_, item) in enumerate(good)
            ]

        results, seconds = timed(post)
        stages["postprocess"] += seconds
        for (path, _), result in zip(good, results):
            detections[str(path)] = result
    stages["staged_total"] = sum(stages.values())
    stages["wall_total"] = time.perf_counter() - total_start
    return {"stages": stages, "detections": detections}


def run_end_to_end(detector: FastOnnxNudeDetector, paths: list[Path], batch_size: int) -> float:
    """Wall time of the real overlapped pipeline (prepare N+1 while running N)."""
    start = time.perf_counter()
    batches = list(chunks(paths, batch_size))
    prepared = detector.prepare_batch(batches[0]) if batches else None
    for index in range(len(batches)):
        following = (
            detector.prepare_batch(batches[index + 1]) if index + 1 < len(batches) else None
        )
        detector.run_prepared(prepared)
        prepared = following
    return time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dir", type=Path)
    source.add_argument("--synthetic", type=int, metavar="N")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="cpu")
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--fast-decode", action="store_true")
    parser.add_argument("--nude-threshold", type=float, default=0.55)
    parser.add_argument("--sexy-threshold", type=float, default=0.55)
    parser.add_argument("--save-golden", type=Path)
    parser.add_argument("--compare-golden", type=Path)
    args = parser.parse_args(argv)

    temp = None
    if args.synthetic:
        temp = tempfile.TemporaryDirectory()
        paths = make_synthetic(Path(temp.name), args.synthetic)
    else:
        paths = list(iter_images(args.dir, args.dir / "_classified"))[: args.limit]
    if not paths:
        print("No images found.")
        return 1

    providers = get_onnx_providers(args.device)
    detector = FastOnnxNudeDetector(
        providers=providers,
        preprocess_workers=args.preprocess_workers,
        fast_decode=args.fast_decode,
    )
    try:
        print(f"images={len(paths)} batch={args.batch_size} providers={detector.providers} "
              f"workers={args.preprocess_workers} fast_decode={args.fast_decode}")
        run_bench(detector, paths[: args.batch_size], args.batch_size)  # warm-up
        report = run_bench(detector, paths, args.batch_size)
        stages = report["stages"]
        count = len(paths)
        for name in ("preprocess", "inference", "postprocess"):
            print(f"{name:<12}{rate(count, stages[name])}")
        print(f"{'sequential':<12}{rate(count, stages['wall_total'])}")
        overlapped = run_end_to_end(detector, paths, args.batch_size)
        print(f"{'overlapped':<12}{rate(count, overlapped)}")
    finally:
        detector.close()

    categories = {
        path: classify_detection(result, args.nude_threshold, args.sexy_threshold)[0]
        for path, result in report["detections"].items()
    }
    if args.save_golden:
        args.save_golden.write_text(json.dumps(categories, indent=1), encoding="utf-8")
        print(f"golden saved: {args.save_golden} ({len(categories)} images)")
    exit_code = 0
    if args.compare_golden:
        golden = json.loads(args.compare_golden.read_text(encoding="utf-8"))
        common = set(golden) & set(categories)
        changed = sorted(p for p in common if golden[p] != categories[p])
        agreement = 1 - len(changed) / len(common) if common else 0.0
        print(f"golden agreement: {agreement:.2%} ({len(changed)} changed of {len(common)})")
        for path in changed[:10]:
            print(f"  {path}: {golden[path]} -> {categories[path]}")
        if agreement < 0.99:
            exit_code = 2
    if temp:
        temp.cleanup()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
