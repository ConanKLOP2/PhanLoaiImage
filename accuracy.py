"""Measure and tune classification accuracy against hand-labelled images.

Workflow: score_images.py (raw model scores) -> make_sample.py (pick images to
label) -> label_tool.py (label by hand) -> evaluate.py (metrics + threshold
sweep). Everything here works on cached raw scores, so thresholds and rules can
be tried offline without running the model again.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from policy import classify_detection

CLASSES = ("nude", "sexy", "normal")
SEVERITY = {"normal": 0, "sexy": 1, "nude": 2}


# --- files -----------------------------------------------------------------


def read_scores(path: Path) -> dict[str, list[dict]]:
    """source -> detections, skipping rows whose image failed to decode."""
    scores: dict[str, list[dict]] = {}
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            scores[row["source"]] = row["detections"]
    return scores


def read_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("label") in CLASSES:
                labels[row["path"]] = row["label"]
    return labels


def read_sample(path: Path) -> list["SampleRow"]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return [
            SampleRow(row["path"], row["stratum"], float(row["weight"]))
            for row in csv.DictReader(file)
        ]


def write_sample(path: Path, rows: Iterable["SampleRow"]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "stratum", "weight"])
        for row in rows:
            writer.writerow([row.path, row.stratum, f"{row.weight:.6f}"])


# --- sampling --------------------------------------------------------------


@dataclass(frozen=True)
class SampleRow:
    path: str
    stratum: str  # class the model predicted at sampling time
    weight: float  # how many images of the population this one stands for


def select_sample(
    scores: dict[str, list[dict]],
    per_class: int,
    nude_threshold: float = 0.55,
    sexy_threshold: float = 0.55,
    seed: int = 0,
) -> list[SampleRow]:
    """Stratified sample by predicted class.

    Rare classes get as many images as common ones, so every class has enough
    examples. Each row carries a weight (stratum size / images drawn) so the
    metrics still estimate the whole folder, not the sample.
    """
    strata: dict[str, list[str]] = {name: [] for name in CLASSES}
    for source, detections in scores.items():
        label, _ = classify_detection(detections, nude_threshold, sexy_threshold)
        strata[label].append(source)

    rng = random.Random(seed)
    rows: list[SampleRow] = []
    for name in CLASSES:
        population = sorted(strata[name])
        if not population:
            continue
        chosen = rng.sample(population, min(per_class, len(population)))
        weight = len(population) / len(chosen)
        rows.extend(SampleRow(path, name, weight) for path in sorted(chosen))
    rng.shuffle(rows)
    return rows


# --- metrics ---------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    path: str
    label: str
    detections: list[dict] = field(repr=False)
    weight: float = 1.0


def build_items(
    scores: dict[str, list[dict]],
    labels: dict[str, str],
    sample: list[SampleRow] | None = None,
) -> list[Item]:
    weights = {row.path: row.weight for row in sample or []}
    return [
        Item(path, label, scores[path], weights.get(path, 1.0))
        for path, label in labels.items()
        if path in scores
    ]


@dataclass(frozen=True)
class ClassStats:
    precision: float
    recall: float
    f1: float
    support: float


@dataclass(frozen=True)
class Metrics:
    confusion: dict[tuple[str, str], float]  # (true, predicted) -> weight
    per_class: dict[str, ClassStats]
    accuracy: float
    macro_f1: float
    total: float


def predict(item: Item, nude_threshold: float, sexy_threshold: float) -> str:
    return classify_detection(item.detections, nude_threshold, sexy_threshold)[0]


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    items: list[Item],
    nude_threshold: float,
    sexy_threshold: float,
    predictor: Callable[[Item], str] | None = None,
) -> Metrics:
    confusion = {(t, p): 0.0 for t in CLASSES for p in CLASSES}
    for item in items:
        predicted = (
            predictor(item) if predictor else predict(item, nude_threshold, sexy_threshold)
        )
        confusion[(item.label, predicted)] += item.weight

    total = sum(confusion.values())
    per_class: dict[str, ClassStats] = {}
    for name in CLASSES:
        true_positive = confusion[(name, name)]
        predicted_total = sum(confusion[(t, name)] for t in CLASSES)
        actual_total = sum(confusion[(name, p)] for p in CLASSES)
        precision = _ratio(true_positive, predicted_total)
        recall = _ratio(true_positive, actual_total)
        per_class[name] = ClassStats(
            precision, recall, _ratio(2 * precision * recall, precision + recall), actual_total
        )
    accuracy = _ratio(sum(confusion[(n, n)] for n in CLASSES), total)
    # Macro-F1 averages only classes that have labelled examples; a class with no
    # ground truth would otherwise drag the score down for something unmeasurable.
    measured = [stats.f1 for stats in per_class.values() if stats.support > 0]
    macro_f1 = sum(measured) / len(measured) if measured else 0.0
    return Metrics(confusion, per_class, accuracy, macro_f1, total)


def unmeasured_classes(items: list[Item]) -> list[str]:
    """Classes with no labelled example: their recall and thresholds are unverified."""
    present = {item.label for item in items}
    return [name for name in CLASSES if name not in present]


def mistakes(
    items: list[Item], nude_threshold: float, sexy_threshold: float
) -> list[tuple[Item, str]]:
    """Items the classifier gets wrong, with what it predicted."""
    wrong = []
    for item in items:
        predicted = predict(item, nude_threshold, sexy_threshold)
        if predicted != item.label:
            wrong.append((item, predicted))
    return wrong


# --- objectives and threshold search ---------------------------------------

Objective = Callable[[Metrics], float]


def macro_f1_objective(metrics: Metrics) -> float:
    return metrics.macro_f1


def cost_objective(miss_cost: float = 3.0, false_alarm_cost: float = 1.0) -> Objective:
    """Negative mean cost. Under-rating an image (nude -> normal) costs miss_cost,
    over-rating it (normal -> nude) costs false_alarm_cost, per step of severity."""

    def objective(metrics: Metrics) -> float:
        total_cost = 0.0
        for (true, predicted), weight in metrics.confusion.items():
            gap = SEVERITY[predicted] - SEVERITY[true]
            if gap < 0:
                total_cost += weight * miss_cost * -gap
            elif gap > 0:
                total_cost += weight * false_alarm_cost * gap
        return -_ratio(total_cost, metrics.total)

    return objective


def threshold_grid(low: float = 0.1, high: float = 0.95, step: float = 0.05) -> list[float]:
    count = round((high - low) / step)
    return [round(low + index * step, 4) for index in range(count + 1)]


@dataclass(frozen=True)
class SweepResult:
    nude_threshold: float
    sexy_threshold: float
    score: float
    metrics: Metrics


def sweep(
    items: list[Item],
    grid: list[float] | None = None,
    objective: Objective = macro_f1_objective,
) -> list[SweepResult]:
    """Every (nude, sexy) pair, best first. Ties prefer the lower thresholds
    only through sort stability on the grid order, so results are deterministic."""
    grid = grid or threshold_grid()
    results = []
    for nude_threshold in grid:
        for sexy_threshold in grid:
            metrics = evaluate(items, nude_threshold, sexy_threshold)
            results.append(
                SweepResult(nude_threshold, sexy_threshold, objective(metrics), metrics)
            )
    results.sort(key=lambda result: -result.score)
    return results


@dataclass(frozen=True)
class FoldResult:
    nude_threshold: float
    sexy_threshold: float
    train_score: float
    held_out_score: float


def cross_validate(
    items: list[Item],
    folds: int = 5,
    grid: list[float] | None = None,
    objective: Objective = macro_f1_objective,
    seed: int = 0,
) -> list[FoldResult]:
    """Pick thresholds on k-1 folds, score them on the held-out fold.

    The held-out score is an honest estimate; the best sweep score is optimistic
    because it was chosen on the same images it is measured on.
    """
    if folds < 2:
        raise ValueError("folds phai >= 2")
    if len(items) < folds:
        raise ValueError("Khong du anh da gan nhan de chia folds")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    parts = [shuffled[index::folds] for index in range(folds)]
    results = []
    for held_index, held_out in enumerate(parts):
        training = [
            item for index, part in enumerate(parts) if index != held_index for item in part
        ]
        best = sweep(training, grid, objective)[0]
        held_score = objective(evaluate(held_out, best.nude_threshold, best.sexy_threshold))
        results.append(
            FoldResult(best.nude_threshold, best.sexy_threshold, best.score, held_score)
        )
    return results
