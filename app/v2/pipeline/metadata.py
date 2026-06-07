"""
Metadata fingerprinting module.
Extracts structural signals from each PDF page and scores boundary likelihood
between consecutive page pairs using four complementary signals.
"""

import fitz


def fingerprint(page: fitz.Page) -> dict:
    """
    Extract a structural fingerprint from a single fitz.Page.

    Returns a dict with:
        size:         (width_pt, height_pt) rounded to nearest integer
        fonts:        set of basefont name strings on this page
        image_count:  number of embedded image XObjects
        text_length:  character count of the embedded text layer
    """
    return {
        "size": (round(page.rect.width), round(page.rect.height)),
        "fonts": set(f[3] for f in page.get_fonts()),
        "image_count": len(page.get_images()),
        "text_length": len(page.get_text()),
    }


def boundary_score(fp1: dict, fp2: dict) -> int:
    """
    Compute a boundary score (0–8) comparing two consecutive page fingerprints.
    Higher score = stronger evidence that a document boundary exists between them.

    Scoring:
        Page size change:              +3  (strong — different doc formats)
        Font set completely different: +2  (strong for digital PDFs)
        Font set partially different:  +1  (moderate — partial template change)
        Image count delta > 1:         +1  (weak alone, strong combined)
        Text length delta > 500 chars: +2  (moderate — content type change)

    Scanned pages return empty font sets — font signal is skipped in that case.
    """
    score = 0

    # Signal 1: page size
    if fp1["size"] != fp2["size"]:
        score += 3

    # Signal 2: font set change
    f1, f2 = fp1["fonts"], fp2["fonts"]
    if f1 and f2:
        overlap = f1 & f2
        if len(overlap) == 0:
            score += 2  # completely different font stack
        elif len(overlap) < min(len(f1), len(f2)) / 2:
            score += 1  # partial change

    # Signal 3: image count spike
    if abs(fp1["image_count"] - fp2["image_count"]) > 1:
        score += 1

    # Signal 4: text length dramatic change
    if abs(fp1["text_length"] - fp2["text_length"]) > 500:
        score += 2

    return score


def fingerprint_all_pages(doc: fitz.Document) -> list[dict]:
    """
    Compute fingerprints for all pages in a document.

    Args:
        doc: Open fitz.Document (must not be closed).

    Returns:
        List of fingerprint dicts, one per page, 0-indexed.
    """
    return [fingerprint(doc[i]) for i in range(len(doc))]
