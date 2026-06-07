"""
Boundary detection module.
Combines metadata fingerprint scores, perceptual hash Hamming distances,
and heading text confirmation to detect document boundaries within a PDF.
Also handles post-classification segment merging.
"""

import fitz
from PIL import Image

from app.v2.pipeline.metadata import fingerprint_all_pages, boundary_score
from app.v2.pipeline.phash import compute_phash_all_pages, hamming_distance


# ── Boundary decision thresholds ────────────────────────────────────────────

def is_probable_boundary(meta_score: int, hamming: int) -> bool:
    """
    Returns True if metadata + pHash signals together suggest a boundary.
    This is a PROBABLE boundary — heading confirmation runs next.

    Rules (any one sufficient):
        Strong metadata (5+) + any visual difference (10+)
        Moderate metadata (3+) + moderate visual difference (15+)
        Weak metadata (<3)  + strong visual difference (21+)
    """
    if meta_score >= 5 and hamming >= 10:
        return True
    if meta_score >= 3 and hamming >= 15:
        return True
    if meta_score < 3 and hamming >= 21:
        return True
    return False


# ── Heading confirmation ─────────────────────────────────────────────────────

def get_first_heading(page: fitz.Page) -> str:
    """
    Extract the first non-empty text line from a page's embedded text layer.
    Returns empty string if the page has no text layer (scanned page).
    """
    text = page.get_text().strip()
    if not text:
        return ""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped.lower()
    return ""


def confirm_boundary_with_heading(
    page_before: fitz.Page,
    page_after: fitz.Page,
) -> tuple[bool, str | None]:
    """
    Use first-line heading comparison to confirm or suppress a probable boundary.

    Returns:
        (confirmed: bool, result_label: str | None)
        confirmed=True  → boundary confirmed (keep it)
        confirmed=False → boundary suppressed (merge pages into same segment)
        result_label:   "confirmed_heading_change" |
                        "suppressed_same_heading"  |
                        "skipped_no_text_layer"
    """
    h_before = get_first_heading(page_before)
    h_after  = get_first_heading(page_after)

    if not h_before or not h_after:
        # Cannot do semantic check — trust pHash + metadata
        return True, "skipped_no_text_layer"

    if h_before == h_after:
        # Same heading = same document (false positive suppressed)
        return False, "suppressed_same_heading"

    # Different headings = real boundary confirmed
    return True, "confirmed_heading_change"


# ── Confidence scoring ───────────────────────────────────────────────────────

def _confidence_band(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.50:
        return "MEDIUM"
    return "LOW"


def boundary_confidence_score(
    meta_score: int,
    hamming: int,
    heading_result: str | None,
) -> dict:
    """
    Compute a normalised boundary confidence score (0.0–1.0) and reasoning string.

    Components:
        meta_component    = min(meta_score / 8, 1.0)
        hamming_component = min(hamming / 40, 1.0)
        heading_bonus:
            +0.10 if heading confirmed change
            -0.15 if heading suppressed (same heading)
             0.00 if heading skipped (no text layer)

    Final = clamp(0.5 * meta + 0.5 * hamming + bonus, 0.0, 1.0)
    """
    meta_c    = min(meta_score / 8, 1.0)
    hamming_c = min(hamming / 40, 1.0)

    bonus_map = {
        "confirmed_heading_change": 0.10,
        "suppressed_same_heading":  -0.15,
        "skipped_no_text_layer":    0.0,
        None:                       0.0,
    }
    bonus = bonus_map.get(heading_result, 0.0)
    raw   = 0.5 * meta_c + 0.5 * hamming_c + bonus
    score = round(max(0.0, min(1.0, raw)), 3)

    # Reasoning string
    parts = []
    parts.append(
        f"metadata score {meta_score}/8 "
        f"({'strong' if meta_score >= 5 else 'moderate' if meta_score >= 3 else 'weak'})"
    )
    parts.append(
        f"pHash Hamming={hamming} "
        f"({'high' if hamming >= 21 else 'moderate' if hamming >= 11 else 'low'} visual divergence)"
    )
    if heading_result == "confirmed_heading_change":
        parts.append("heading text changed — boundary confirmed")
    elif heading_result == "suppressed_same_heading":
        parts.append("heading text unchanged — false positive suppressed")
    else:
        parts.append("heading check skipped (no text layer)")

    return {
        "confidence":       score,
        "confidence_band":  _confidence_band(score),
        "reasoning":        "; ".join(parts),
    }


# ── Main boundary detection ──────────────────────────────────────────────────

def detect_boundaries(
    doc: fitz.Document,
    page_images: list[Image.Image],
) -> list[dict]:
    """
    Detect document boundaries within a PDF and return a list of segments.
    Each segment covers a contiguous range of pages belonging to one document.

    Args:
        doc:         Open fitz.Document (must not be closed during this call).
        page_images: List of PIL Images, 0-indexed, from split_pdf().

    Returns:
        List of segment dicts:
        {
            "page_range":          [start_0based, end_0based],
            "representative_image": PIL.Image (first page of segment),
            "boundary_info":        dict | None (None for first segment),
            "boundaries_suppressed": int (count of suppressed boundaries before this segment)
        }
    """
    n = len(page_images)

    if n == 1:
        return [{
            "page_range":           [0, 0],
            "representative_image": page_images[0],
            "boundary_info":        None,
            "boundaries_suppressed": 0,
        }]

    # Precompute fingerprints and hashes for all pages
    fingerprints = fingerprint_all_pages(doc)
    hashes       = compute_phash_all_pages(page_images)

    # Scan consecutive pairs for boundaries
    confirmed_boundaries  = []   # (page_index_after_boundary, boundary_info)
    total_suppressed      = 0

    for i in range(n - 1):
        meta_s  = boundary_score(fingerprints[i], fingerprints[i + 1])
        hamming = hamming_distance(hashes[i], hashes[i + 1])

        if is_probable_boundary(meta_s, hamming):
            confirmed, heading_result = confirm_boundary_with_heading(
                doc[i], doc[i + 1]
            )
            b_info = boundary_confidence_score(meta_s, hamming, heading_result)

            if confirmed:
                confirmed_boundaries.append((i + 1, b_info))
            else:
                total_suppressed += 1
        # else: clearly same document — no further check needed

    # Build segments from confirmed boundary positions
    segments = []
    start = 0

    boundary_positions = [pos for pos, _ in confirmed_boundaries]
    boundary_infos     = {pos: info for pos, info in confirmed_boundaries}

    for boundary_pos in boundary_positions:
        segments.append({
            "page_range":           [start, boundary_pos - 1],
            "representative_image": page_images[start],
            "boundary_info":        None if start == 0 else boundary_infos.get(start),
            "boundaries_suppressed": total_suppressed if start == 0 else 0,
        })
        start = boundary_pos

    # Final segment
    segments.append({
        "page_range":           [start, n - 1],
        "representative_image": page_images[start],
        "boundary_info":        boundary_infos.get(start),
        "boundaries_suppressed": 0,
    })

    return segments, total_suppressed


# ── Post-classification merge ────────────────────────────────────────────────

def merge_same_class_segments(classified_segments: list[dict]) -> list[dict]:
    """
    After SigLIP 2 classification, merge adjacent segments that resolve to
    the same class AND the same top hypothesis (e.g. Aadhaar front + back).

    Keeps segments separate if they share a class but different top hypothesis
    (e.g. PAN card + Aadhaar both classify as kyc_document but via different
    hypothesis anchors — they are two distinct physical documents).

    Args:
        classified_segments: List of segment dicts with classification results.

    Returns:
        Merged list of segment dicts.
    """
    if len(classified_segments) <= 1:
        return classified_segments

    merged = [classified_segments[0].copy()]

    for current in classified_segments[1:]:
        prev = merged[-1]
        same_class      = prev["predicted_class"] == current["predicted_class"]
        same_hypothesis = prev["top_hypothesis"]  == current["top_hypothesis"]

        if same_class and same_hypothesis:
            # Extend page range of previous segment
            prev["page_range"][1] = current["page_range"][1]
        else:
            merged.append(current.copy())

    return merged
