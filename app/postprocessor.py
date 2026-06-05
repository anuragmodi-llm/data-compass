from __future__ import annotations

from typing import Any

from classifier import CategoryTree

WARNING_LOW_CONFIDENCE = "Low confidence — recommend manual review"

_BAND_HIGH = 70.0
_BAND_MEDIUM = 40.0


def _confidence_band(pct: float) -> str:
    if pct >= _BAND_HIGH:
        return "HIGH"
    if pct >= _BAND_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _score_to_pct(score: float) -> float:
    return round(score * 100, 1)


def _format_score_entry(
    label: str,
    score: float,
    passed_gate: bool,
    label_to_display: dict[str, str],
) -> dict[str, Any]:
    return {
        "label": label,
        "display_name": label_to_display.get(label, label),
        "score": _score_to_pct(score),
        "passed_gate": passed_gate,
    }


def _apply_gate(
    scores: dict[str, float],
    tree: CategoryTree,
) -> tuple[str, float, bool]:
    """
    Returns (winning_label, winning_score_raw, passed_gate).
    """
    threshold = tree.default_threshold
    passing = {lbl: s for lbl, s in scores.items() if s >= threshold}

    if not passing:
        reject_score = scores.get(tree.reject_label, 0.0)
        return tree.reject_label, reject_score, False

    winner = max(passing, key=lambda lbl: passing[lbl])
    return winner, passing[winner], True


def _format_page_result(
    page: dict[str, Any],
    tree: CategoryTree,
    label_to_display: dict[str, str],
) -> dict[str, Any]:
    raw_scores: dict[str, float] = page["scores"]
    threshold = tree.default_threshold

    all_scores = sorted(
        [
            _format_score_entry(lbl, s, s >= threshold, label_to_display)
            for lbl, s in raw_scores.items()
        ],
        key=lambda e: e["score"],
        reverse=True,
    )

    winner_label = page["winner"]
    winner_score_pct = _score_to_pct(page["score"])
    passed = page["passed_gate"]
    band = _confidence_band(winner_score_pct)

    return {
        "page": page["page"],
        "predicted_label": winner_label,
        "display_name": label_to_display.get(winner_label, winner_label),
        "confidence": winner_score_pct,
        "status": "PASS" if passed else "REJECT",
        "confidence_band": band,
        "all_scores": all_scores,
        "warning": WARNING_LOW_CONFIDENCE if band == "LOW" else None,
    }


def build_response(classify_result: dict[str, Any], tree: CategoryTree) -> dict[str, Any]:
    """
    Convert the raw output of classifier.classify() into the final API response.

    Args:
        classify_result: dict returned by classifier.classify()
        tree:            CategoryTree (for gate config and display names)

    Returns:
        Structured response dict matching the specified output schema.
    """
    label_to_display: dict[str, str] = {
        cat.label: cat.display_name for cat in tree.categories
    }

    winning_label: str = classify_result["label"]
    winning_score_raw: float = classify_result["score"]
    passed_gate: bool = classify_result["passed_gate"]
    winning_pct = _score_to_pct(winning_score_raw)
    band = _confidence_band(winning_pct)

    # Build all_scores from best page's raw scores
    best_page_idx = classify_result["best_page"] - 1
    best_page_data = classify_result["pages"][best_page_idx]
    raw_scores: dict[str, float] = best_page_data["scores"]
    threshold = tree.default_threshold

    all_scores = sorted(
        [
            _format_score_entry(lbl, s, s >= threshold, label_to_display)
            for lbl, s in raw_scores.items()
        ],
        key=lambda e: e["score"],
        reverse=True,
    )

    # Per-page breakdown (only included when document has more than one page)
    pages = classify_result["pages"]
    page_results = (
        [_format_page_result(p, tree, label_to_display) for p in pages]
        if classify_result["page_count"] > 1
        else None
    )

    return {
        "predicted_label": winning_label,
        "display_name": label_to_display.get(winning_label, winning_label),
        "confidence": winning_pct,
        "status": "PASS" if passed_gate else "REJECT",
        "confidence_band": band,
        "all_scores": all_scores,
        "page_results": page_results,
        "warning": WARNING_LOW_CONFIDENCE if band == "LOW" else None,
    }
