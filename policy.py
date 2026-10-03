from __future__ import annotations

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
