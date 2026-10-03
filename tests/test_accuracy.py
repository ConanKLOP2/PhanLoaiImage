from __future__ import annotations

import json

import pytest

import accuracy
from accuracy import (
    Item,
    SampleRow,
    build_items,
    cost_objective,
    cross_validate,
    evaluate,
    macro_f1_objective,
    mistakes,
    read_labels,
    read_sample,
    read_scores,
    select_sample,
    sweep,
    threshold_grid,
    write_sample,
)


def det(label, score):
    return [{"class": label, "score": score, "box": [0, 0, 1, 1]}]


def nude(score):
    return det("ANUS_EXPOSED", score)


def sexy(score):
    return det("BELLY_EXPOSED", score)


def hand_items(weight_false_alarm=1.0):
    return [
        Item("a", "nude", nude(0.9)),  # nude  -> nude
        Item("b", "nude", nude(0.3)),  # nude  -> normal (missed)
        Item("c", "sexy", sexy(0.8)),  # sexy  -> sexy
        Item("d", "normal", []),  # normal -> normal
        Item("e", "normal", sexy(0.7), weight_false_alarm),  # normal -> sexy
    ]


# --- metrics ----------------------------------------------------------------


def test_metrics_match_hand_computation():
    metrics = evaluate(hand_items(), 0.5, 0.5)

    assert metrics.confusion[("nude", "nude")] == 1
    assert metrics.confusion[("nude", "normal")] == 1
    assert metrics.confusion[("normal", "sexy")] == 1
    assert metrics.total == 5
    assert metrics.accuracy == pytest.approx(3 / 5)
    nude_stats = metrics.per_class["nude"]
    assert (nude_stats.precision, nude_stats.recall) == (1.0, 0.5)
    assert nude_stats.f1 == pytest.approx(2 / 3)
    assert metrics.per_class["sexy"].precision == pytest.approx(0.5)
    assert metrics.per_class["sexy"].recall == 1.0
    assert metrics.per_class["normal"].f1 == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx((2 / 3 + 2 / 3 + 0.5) / 3)


def test_weights_scale_the_confusion_matrix():
    metrics = evaluate(hand_items(weight_false_alarm=3.0), 0.5, 0.5)

    assert metrics.confusion[("normal", "sexy")] == 3
    assert metrics.total == 7
    assert metrics.per_class["sexy"].precision == pytest.approx(1 / 4)


def test_class_with_no_support_or_predictions_scores_zero_without_dividing_by_zero():
    metrics = evaluate([Item("a", "normal", [])], 0.5, 0.5)

    assert metrics.per_class["nude"].f1 == 0.0
    assert metrics.per_class["nude"].support == 0
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0  # only the class that has examples counts


def test_empty_items_do_not_crash():
    metrics = evaluate([], 0.5, 0.5)

    assert metrics.total == 0 and metrics.accuracy == 0.0 and metrics.macro_f1 == 0.0


def test_thresholds_change_predictions():
    item = Item("a", "nude", nude(0.6))

    assert evaluate([item], 0.5, 0.5).per_class["nude"].recall == 1.0
    assert evaluate([item], 0.7, 0.5).per_class["nude"].recall == 0.0


def test_mistakes_lists_wrong_items_with_prediction():
    wrong = mistakes(hand_items(), 0.5, 0.5)

    assert {(item.path, predicted) for item, predicted in wrong} == {("b", "normal"), ("e", "sexy")}


def test_custom_predictor_overrides_thresholds():
    metrics = evaluate(hand_items(), 0.5, 0.5, predictor=lambda item: item.label)

    assert metrics.accuracy == 1.0 and metrics.macro_f1 == 1.0


# --- objectives --------------------------------------------------------------


def test_cost_objective_penalises_misses_more_than_false_alarms():
    metrics = evaluate(hand_items(), 0.5, 0.5)

    # nude->normal misses 2 steps (3 each = 6); normal->sexy over-rates 1 step (1).
    assert cost_objective(3.0, 1.0)(metrics) == pytest.approx(-(6 + 1) / 5)
    assert cost_objective(1.0, 1.0)(metrics) == pytest.approx(-(2 + 1) / 5)


def test_cost_objective_is_zero_for_perfect_predictions():
    metrics = evaluate(hand_items(), 0.5, 0.5, predictor=lambda item: item.label)

    assert cost_objective()(metrics) == 0.0


def test_higher_miss_cost_lowers_the_chosen_threshold():
    items = [Item(f"n{i}", "nude", nude(0.6 + 0.02 * i)) for i in range(5)]
    items += [Item(f"x{i}", "normal", nude(0.45 + 0.02 * i)) for i in range(5)]
    items += [Item("n_low", "nude", nude(0.4))]

    cheap = sweep(items, objective=cost_objective(1.0, 1.0))[0]
    careful = sweep(items, objective=cost_objective(10.0, 1.0))[0]

    assert careful.nude_threshold <= cheap.nude_threshold


# --- sweep / grid ------------------------------------------------------------


def test_threshold_grid_default_and_custom():
    grid = threshold_grid()

    assert grid[0] == 0.1 and grid[-1] == 0.95 and len(grid) == 18
    assert threshold_grid(0.2, 0.6, 0.2) == [0.2, 0.4, 0.6]


def separable_items():
    items = [Item(f"n{i}", "nude", nude(0.7 + 0.05 * i)) for i in range(4)]
    items += [Item(f"x{i}", "normal", nude(0.2 + 0.05 * i)) for i in range(4)]
    return items


def test_sweep_finds_threshold_that_separates_the_classes():
    results = sweep(separable_items())
    best = results[0]

    assert best.score >= results[1].score
    assert 0.35 <= best.nude_threshold <= 0.7
    assert best.metrics.per_class["nude"].f1 == 1.0
    assert results[-1].score < best.score  # the worst pair is genuinely worse
    assert len(results) == len(threshold_grid()) ** 2


def test_sweep_is_sorted_best_first_and_deterministic():
    first = sweep(separable_items())
    second = sweep(separable_items())

    assert [r.score for r in first] == sorted((r.score for r in first), reverse=True)
    assert [(r.nude_threshold, r.sexy_threshold) for r in first] == [
        (r.nude_threshold, r.sexy_threshold) for r in second
    ]


def test_sweep_uses_the_supplied_objective():
    items = separable_items()

    by_f1 = sweep(items, objective=macro_f1_objective)[0].score
    by_cost = sweep(items, objective=cost_objective())[0].score

    assert by_f1 == pytest.approx(1.0) and by_cost == 0.0  # sexy has no examples, so it is not averaged in


# --- cross validation --------------------------------------------------------


def test_cross_validation_reports_held_out_scores():
    folds = cross_validate(separable_items(), folds=4, seed=1)

    assert len(folds) == 4
    assert all(0.0 <= fold.held_out_score <= 1.0 for fold in folds)


def test_cross_validation_is_seeded():
    items = separable_items()

    assert cross_validate(items, 4, seed=3) == cross_validate(items, 4, seed=3)


def test_cross_validation_validates_inputs():
    with pytest.raises(ValueError):
        cross_validate(separable_items(), folds=1)
    with pytest.raises(ValueError):
        cross_validate(separable_items()[:2], folds=5)


def test_cross_validation_held_out_is_not_better_than_overfit_best():
    # Noisy labels: the in-sample optimum is optimistic compared to held-out.
    items = []
    for index in range(40):
        score = 0.05 + 0.9 * (index / 39)
        label = "nude" if (index % 7 == 0) != (score > 0.5) else "normal"
        items.append(Item(f"i{index}", label, nude(score)))

    best = sweep(items)[0].score
    held = [f.held_out_score for f in cross_validate(items, folds=5, seed=0)]

    assert sum(held) / len(held) <= best + 1e-9


# --- sampling ----------------------------------------------------------------


def population():
    scores = {f"normal{i}": [] for i in range(30)}
    scores.update({f"nude{i}": nude(0.9) for i in range(10)})
    scores.update({f"sexy{i}": sexy(0.9) for i in range(4)})
    return scores


def test_sample_is_stratified_with_weights_that_rebuild_the_population():
    rows = select_sample(population(), per_class=5, seed=0)

    by_stratum = {s: [r for r in rows if r.stratum == s] for s in ("nude", "sexy", "normal")}
    assert {s: len(r) for s, r in by_stratum.items()} == {"nude": 5, "sexy": 4, "normal": 5}
    assert by_stratum["nude"][0].weight == 2.0
    assert by_stratum["sexy"][0].weight == 1.0  # small stratum taken whole
    assert by_stratum["normal"][0].weight == 6.0
    assert sum(r.weight for r in rows) == pytest.approx(44)
    assert len({r.path for r in rows}) == len(rows)


def test_sample_is_deterministic_per_seed_and_varies_across_seeds():
    first = select_sample(population(), 5, seed=1)

    assert first == select_sample(population(), 5, seed=1)
    assert first != select_sample(population(), 5, seed=2)


def test_sample_respects_thresholds_for_strata():
    scores = {"a": nude(0.6), "b": nude(0.6)}

    low = select_sample(scores, 5, nude_threshold=0.5)
    high = select_sample(scores, 5, nude_threshold=0.9, sexy_threshold=0.9)

    assert {r.stratum for r in low} == {"nude"}
    assert {r.stratum for r in high} == {"normal"}


def test_sample_with_empty_scores_is_empty():
    assert select_sample({}, 5) == []


def test_sample_recovers_population_accuracy_estimate():
    # 90 normal images all correct, 10 nude images half missed. Weighted accuracy
    # from a small stratified sample should land near the population value.
    scores = {f"n{i}": [] for i in range(90)}
    scores.update({f"x{i}": nude(0.9) for i in range(10)})
    rows = select_sample(scores, per_class=10, seed=0)
    labels = {r.path: ("nude" if r.path.startswith("x") else "normal") for r in rows}
    items = build_items(scores, labels, rows)

    assert evaluate(items, 0.5, 0.5).accuracy == pytest.approx(1.0)
    assert evaluate(items, 0.5, 0.5).total == pytest.approx(100)


# --- files -------------------------------------------------------------------


def write_scores(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_read_scores_skips_errors_and_blank_lines(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        json.dumps({"source": "a", "detections": nude(0.9), "error": None}) + "\n\n"
        + json.dumps({"source": "bad", "detections": [], "error": "ValueError"}) + "\n",
        encoding="utf-8",
    )

    assert list(read_scores(path)) == ["a"]


def test_read_labels_ignores_unknown_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("path,label\na,nude\nb,weird\nc,normal\n", encoding="utf-8")

    assert read_labels(path) == {"a": "nude", "c": "normal"}


def test_sample_file_round_trip_with_awkward_paths(tmp_path):
    path = tmp_path / "sample.csv"
    rows = [SampleRow('C:\\a, "b"\\ảnh.jpg', "nude", 2.5), SampleRow("x.jpg", "normal", 1.0)]

    write_sample(path, rows)

    assert read_sample(path) == rows


def test_build_items_keeps_only_images_with_scores_and_applies_weights():
    scores = {"a": [], "b": nude(0.9)}
    labels = {"a": "normal", "b": "nude", "ghost": "sexy"}
    sample = [SampleRow("b", "nude", 4.0)]

    items = build_items(scores, labels, sample)

    assert {i.path: i.weight for i in items} == {"a": 1.0, "b": 4.0}


def test_module_exposes_classes_in_severity_order():
    assert accuracy.CLASSES == ("nude", "sexy", "normal")
    assert accuracy.SEVERITY["normal"] < accuracy.SEVERITY["sexy"] < accuracy.SEVERITY["nude"]


def test_unmeasured_classes_lists_classes_without_labels():
    from accuracy import unmeasured_classes

    assert unmeasured_classes(hand_items()) == []
    assert unmeasured_classes([Item("a", "normal", []), Item("b", "sexy", [])]) == ["nude"]
    assert unmeasured_classes([]) == ["nude", "sexy", "normal"]


def test_macro_f1_ignores_class_without_ground_truth_but_penalises_false_alarms():
    # No nude truth at all; two normal images wrongly called nude.
    items = [Item("a", "normal", []), Item("b", "normal", nude(0.9)), Item("c", "normal", nude(0.9))]

    low = evaluate(items, 0.5, 0.5)
    high = evaluate(items, 0.95, 0.95)

    assert low.macro_f1 == pytest.approx(2 * (1 / 3) / (1 + 1 / 3))  # normal: precision 1, recall 1/3
    assert high.macro_f1 == 1.0
    assert sweep(items)[0].nude_threshold >= 0.95 - 1e-9
