"""Pick images to label by hand, stratified by the model's current prediction.

    python make_sample.py --scores scores.jsonl --per-class 100 --out sample.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from accuracy import CLASSES, read_scores, select_sample, write_sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("sample.csv"))
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--nude-threshold", type=float, default=0.55)
    parser.add_argument("--sexy-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    scores = read_scores(args.scores)
    if not scores:
        raise SystemExit("Khong co anh nao trong file scores.")
    rows = select_sample(
        scores, args.per_class, args.nude_threshold, args.sexy_threshold, args.seed
    )
    write_sample(args.out, rows)

    print(f"{len(scores)} anh co diem so -> chon {len(rows)} anh de gan nhan: {args.out}")
    for name in CLASSES:
        picked = [row for row in rows if row.stratum == name]
        if picked:
            print(
                f"  du doan {name:<7}: chon {len(picked):4d} / {round(len(picked) * picked[0].weight):5d}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
