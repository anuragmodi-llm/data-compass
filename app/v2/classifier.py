"""
V2 classifier wrapper.
Reuses the shared SigLIP 2 model loaded by v1 — no second model copy.
Scores each segment's representative image against industry-scoped hypotheses only.
"""

from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

from app.v1.classifier import _ensure_model, _score_image_against_texts

HYPOTHESES_PATH = Path(__file__).parent / "hypotheses.yaml"

_hypotheses_cache: dict | None = None


def _load_hypotheses() -> dict:
    global _hypotheses_cache
    if _hypotheses_cache is None:
        with open(HYPOTHESES_PATH, "r", encoding="utf-8") as fh:
            _hypotheses_cache = yaml.safe_load(fh)
    return _hypotheses_cache


def load_hypotheses_for_industry(industry: Optional[str]) -> list[dict]:
    """
    Load hypothesis entries for a specific industry from hypotheses.yaml.

    Each entry is: {"class": str, "hypothesis": str}

    If industry is None (vLM router returned no match or Phase 2.2 not yet
    implemented), returns ALL hypotheses from every industry plus common entries,
    so classification can still proceed against the full hypothesis set.

    Args:
        industry: Industry label string (e.g. "Banking & Lending"), or None.

    Returns:
        List of hypothesis dicts with "class" and "hypothesis" keys.
    """
    data = _load_hypotheses()
    industries = data.get("industries", {})
    common = data.get("common", [])

    if industry is None:
        all_hyps = []
        for hyp_list in industries.values():
            all_hyps.extend(hyp_list)
        all_hyps.extend(common)
        return all_hyps

    industry_hyps = industries.get(industry, [])
    # common entries (photograph, out_of_category) are always appended as a
    # safety net so scoped classification can still reject non-documents
    return industry_hyps + common


def classify_with_siglip2(image: Image.Image, hypotheses: list[dict]) -> dict:
    """
    Score a PIL image against a list of hypothesis dicts using SigLIP 2.

    Uses max-sigmoid aggregation per class: when a class has multiple hypotheses,
    the highest-scoring hypothesis represents that class score.

    Args:
        image:      PIL Image of the document page (representative page of segment).
        hypotheses: List of {"class": str, "hypothesis": str} dicts,
                    as returned by load_hypotheses_for_industry().

    Returns:
        {
            "class":          str,   # label of winning class
            "top_score":      float, # sigmoid score of winning hypothesis
            "second_score":   float, # sigmoid score of second-best class
            "top_hypothesis": str    # hypothesis text that won
        }
    """
    if not hypotheses:
        return {
            "class": "out_of_category",
            "top_score": 0.0,
            "second_score": 0.0,
            "top_hypothesis": "",
        }

    model, processor = _ensure_model()
    texts = [h["hypothesis"] for h in hypotheses]

    # Score image against all hypothesis texts in one forward pass
    scores = _score_image_against_texts(image, texts, model, processor)

    # Max-sigmoid aggregation: keep best score + winning hypothesis text per class
    class_best: dict[str, tuple[float, str]] = {}
    for hyp, score in zip(hypotheses, scores):
        cls = hyp["class"]
        hyp_text = hyp["hypothesis"]
        if cls not in class_best or score > class_best[cls][0]:
            class_best[cls] = (score, hyp_text)

    ranked = sorted(class_best.items(), key=lambda x: x[1][0], reverse=True)

    top_class, (top_score, top_hyp) = ranked[0]
    second_score = ranked[1][1][0] if len(ranked) > 1 else 0.0

    return {
        "class": top_class,
        "top_score": round(top_score, 4),
        "second_score": round(second_score, 4),
        "top_hypothesis": top_hyp,
    }
