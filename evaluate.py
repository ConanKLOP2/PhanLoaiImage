"""Accuracy report and threshold search against hand labels.

    python evaluate.py --scores scores.jsonl --labels labels.csv --sample sample.csv
    python evaluate.py ... --objective cost --miss-cost 3     # prefer not missing images
    python evaluate.py ... --errors 20                         # list the wrong ones

--sample (optional) supplies stratified-sampling weights so the numbers describe
the whole folder rather than the labelled subset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from accuracy import (
    CLASSES,
    Metrics,
    build_items,
    cost_objective,
    cross_validate,
    evaluate,
    macro_f1_objective,
    mistakes,
    read_labels,
    read_sample,
    read_scores,
    sweep,
    threshold_grid,
    unmeasured_classes,
)


def format_metrics(metrics: Metrics, title: str) -> str:
    lines = [
        title,
        f"  accuracy={metrics.accuracy:.1%}  macro-F1={metrics.macro_f1:.3f}",
        f"  {'':8}{'precision':>10}{'recall':>9}{'F1':>7}{'support':>10}",
    ]
    for name in CLASSES:
        stats = metrics.per_class[name]
        lines.append(
            f"  {name:<8}{stats.precision:>10.1%}{stats.recall:>9.1%}{stats.f1:>7.3f}{stats.support:>10.0f}"
        )
    lines.append("  confusion (rows = truth, columns = predicted):")
    lines.append("  " + " " * 8 + "".join(f"{name:>9}" for name in CLASSES))
    for true in CLASSES:
        lines.append(
            "  "
            + f"{true:<8}"
            + "".join(f"{metrics.confusion[(true, pred)]:>9.0f}" for pred in CLASSES)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=None)
    parser.add_argument("--objective", choices=["macro-f1", "cost"], default="macro-f1")
    parser.add_argument("--miss-cost", type=float, default=3.0, help="cost of rating an image too low, per severity step")
    parser.add_argument("--false-alarm-cost", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--current", type=float, nargs=2, action="append", metavar=("NUDE", "SEXY"),
                        help="thresholds to report as baseline (repeatable); default 0.55 0.55 and 0.8 0.8")
    parser.add_argument("--errors", type=int, default=0, help="list this many misclassified images")
    parser.add_argument("--out", type=Path, default=None, help="write a JSON report")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    scores = read_scores(args.scores)
    labels = read_labels(args.labels)
    sample = read_sample(args.sample) if args.sample else None
    items = build_items(scores, labels, sample)
    if not items:
        raise SystemExit("Khong co anh nao vua co diem so vua co nhan.")

    objective = (
        macro_f1_objective
        if args.objective == "macro-f1"
        else cost_objective(args.miss_cost, args.false_alarm_cost)
    )
    grid = threshold_grid(step=args.step)
    report: dict = {"labelled": len(items), "weighted": sample is not None}

    print(f"{len(items)} anh co nhan" + (" (co trong so lay mau)" if sample else " (khong trong so)"))
    missing = unmeasured_classes(items)
    if missing:
        print(f"CANH BAO: khong co anh that nao thuoc lop {', '.join(missing)}.")
        print("  Recall cua lop do va nguong cua no khong kiem chung duoc;")
        print("  macro-F1 chi tinh cac lop co anh. Chi dung ket qua de danh gia cac lop con lai.")
        report["unmeasured_classes"] = missing
    print()

    baselines = args.current or [[0.55, 0.55], [0.8, 0.8]]
    report["baselines"] = []
    for nude_threshold, sexy_threshold in baselines:
        metrics = evaluate(items, nude_threshold, sexy_threshold)
        print(format_metrics(metrics, f"Hien tai: nude>={nude_threshold} sexy>={sexy_threshold}"))
        print(f"  objective={objective(metrics):.4f}\n")
        report["baselines"].append(
            {"nude": nude_threshold, "sexy": sexy_threshold, "macro_f1": metrics.macro_f1,
             "accuracy": metrics.accuracy, "objective": objective(metrics)}
        )

    results = sweep(items, grid, objective)
    best = results[0]
    print(format_metrics(best.metrics, f"Tot nhat tren tap nhan: nude>={best.nude_threshold} sexy>={best.sexy_threshold}"))
    print(f"  objective={best.score:.4f}  (lac quan: da chon tren chinh cac anh nay)\n")
    print("  Top 5 cap nguong:")
    for result in results[:5]:
        print(f"    nude>={result.nude_threshold:.2f} sexy>={result.sexy_threshold:.2f}  objective={result.score:.4f}")
    report["best"] = {
        "nude": best.nude_threshold, "sexy": best.sexy_threshold,
        "objective": best.score, "macro_f1": best.metrics.macro_f1,
    }

    if args.folds >= 2 and len(items) >= args.folds:
        folds = cross_validate(items, args.folds, grid, objective)
        held = [fold.held_out_score for fold in folds]
        print(f"\n  Kiem tra cheo {args.folds} folds (uoc luong trung thuc):")
        for index, fold in enumerate(folds, start=1):
            print(
                f"    fold {index}: chon nude>={fold.nude_threshold:.2f} sexy>={fold.sexy_threshold:.2f}"
                f"  train={fold.train_score:.4f}  held-out={fold.held_out_score:.4f}"
            )
        print(f"    trung binh held-out = {sum(held) / len(held):.4f}")
        report["cross_validation"] = {"held_out_mean": sum(held) / len(held), "folds": args.folds}
    elif args.folds >= 2:
        print(f"\n  Bo qua kiem tra cheo: can it nhat {args.folds} anh co nhan.")

    if args.errors:
        wrong = mistakes(items, best.nude_threshold, best.sexy_threshold)
        wrong.sort(key=lambda pair: -pair[0].weight)
        print(f"\n  {len(wrong)} anh bi nham voi nguong tot nhat (hien {min(args.errors, len(wrong))}):")
        for item, predicted in wrong[: args.errors]:
            top = max(item.detections, key=lambda d: d["score"], default=None)
            hint = f"{top['class']}:{top['score']:.2f}" if top else "no detections"
            print(f"    truth={item.label:<6} predicted={predicted:<6} top={hint}  {item.path}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nBao cao JSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
