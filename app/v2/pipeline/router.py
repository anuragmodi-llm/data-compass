"""
Industry routing module.
Assigns an industry label to each document segment by analysing the text
content of the segment's first page. Returns top-2 industry candidates.

Path A (keyword): digital PDFs with embedded text layer — fast, CPU only.
Path B (vLM):     image uploads and scanned pages — deferred to Phase 2.2.
"""

import fitz
from PIL import Image


# ── Keyword signal dictionary ────────────────────────────────────────────────

INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "Identity & KYC": [
        "uidai", "aadhaar", "unique identification", "pan card",
        "permanent account number", "voter id", "epic", "driving licence",
        "passport", "republic of india", "ckyc", "central kyc",
        "domicile", "caste certificate", "birth certificate",
        "death certificate", "marriage certificate", "udid",
        "non creamy layer", "ration card", "income certificate",
    ],
    "Banking & Lending": [
        "account statement", "ifsc", "micr", "savings account",
        "current account", "transaction", "debit", "credit", "balance",
        "nach", "ecs", "mandate", "loan sanction", "nbfc",
        "cibil", "experian", "credit score", "dpd", "epfo", "uan",
        "provident fund", "loan application",
    ],
    "Investments & Wealth": [
        "demat", "cdsl", "nsdl", "isin", "depository participant",
        "mutual fund", "nav", "folio", "amfi", "cams", "kfintech",
        "consolidated account statement", "cas", "units held",
    ],
    "Insurance": [
        "insurance", "irdai", "policy number", "sum assured", "premium",
        "nominee", "policyholder", "insured", "idv", "insured declared value",
        "claim", "hospitalisation", "mediclaim", "life insurance",
        "health insurance", "motor insurance", "lic",
    ],
    "Taxation & Compliance": [
        "gstin", "hsn", "sac", "cgst", "sgst", "igst", "gst invoice",
        "form 16", "tds", "tan", "traces", "income tax return", "itr",
        "acknowledgment number", "assessment year", "balance sheet",
        "profit and loss", "auditors report", "annual report",
    ],
    "Employment & HR": [
        "offer letter", "appointment letter", "salary slip", "payslip",
        "ctc", "cost to company", "basic salary", "hra", "pf deduction",
        "relieving letter", "experience letter", "last working day",
        "resignation", "reimbursement", "curriculum vitae", "resume",
        "background verification", "bgv",
    ],
    "Education": [
        "marksheet", "mark sheet", "university", "board of education",
        "degree certificate", "diploma", "semester", "cgpa", "percentage",
        "migration certificate", "transfer certificate", "bar council",
        "medical council", "nmc", "icai", "icsi", "enrolment certificate",
    ],
    "Corporate & Business Registration": [
        "certificate of incorporation", "cin", "mca", "registrar of companies",
        "memorandum of association", "articles of association",
        "udyam", "msme", "fssai", "food safety", "shop and establishment",
        "trade license", "board resolution", "power of attorney",
        "vendor agreement", "purchase order", "master service agreement",
    ],
    "Property & Real Estate": [
        "sale deed", "conveyance deed", "sub-registrar", "stamp duty",
        "survey number", "property", "immovable", "seller", "buyer",
        "rental agreement", "leave and licence", "tenant", "landlord",
        "security deposit", "rent",
    ],
    "Medical & Healthcare": [
        "prescription", "diagnosis", "doctor", "physician", "hospital",
        "discharge summary", "pathology", "lab report", "test results",
        "patient name", "registration number", "mbbs",
    ],
    "Procurement & Finance Operations": [
        "subscription", "billing period", "renewal", "saas", "plan",
        "electricity bill", "water bill", "gas bill", "broadband",
        "utility", "service provider", "units consumed", "meter reading",
    ],
    "Legal": [
        "fir", "court order", "affidavit", "legal notice", "tribunal",
        "case number", "advocate", "notary", "no objection certificate",
        "noc", "to whom it may concern",
    ],
}

ALL_INDUSTRIES = list(INDUSTRY_KEYWORDS.keys())


# ── Helpers ──────────────────────────────────────────────────────────────────

def _confidence_band(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.50:
        return "MEDIUM"
    return "LOW"


def extract_text_top(page: fitz.Page, top_fraction: float = 0.4) -> str:
    """
    Extract text from the top 40% of a page using the embedded text layer.
    Returns empty string if no text layer exists (scanned page).
    """
    r = page.rect
    clip = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + r.height * top_fraction)
    return page.get_text(clip=clip)


# ── Path A: keyword routing ──────────────────────────────────────────────────

def keyword_route(text: str) -> dict:
    """
    Match extracted page text against the industry keyword dictionary.
    Returns top-2 industry labels with scores and a reasoning string.

    If no keywords match at all (score == 0 for all industries),
    industry_1 is returned as None — caller should invoke vlm_route().

    Return structure:
    {
        "industry_1":        str | None,
        "industry_1_score":  float (0.0–1.0, normalised by keyword count),
        "industry_2":        str | None,
        "industry_2_score":  float,
        "routing_confidence": float  (= industry_1_score),
        "confidence_band":   str,
        "reasoning":         str,
        "router_path":       "keyword"
    }
    """
    text_lower = text.lower()
    raw_scores: dict[str, int] = {}
    matched_kws: dict[str, list[str]] = {}

    for industry, keywords in INDUSTRY_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in text_lower]
        raw_scores[industry] = len(hits)
        matched_kws[industry] = hits

    ranked = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
    top1_name, top1_raw = ranked[0]
    top2_name, top2_raw = ranked[1]

    if top1_raw == 0:
        return {
            "industry_1":        None,
            "industry_1_score":  0.0,
            "industry_2":        None,
            "industry_2_score":  0.0,
            "routing_confidence": 0.0,
            "confidence_band":   "LOW",
            "reasoning":         "No industry keywords matched in extracted text.",
            "router_path":       "keyword",
        }

    top1_score = round(top1_raw / len(INDUSTRY_KEYWORDS[top1_name]), 3)
    top2_score = (
        round(top2_raw / len(INDUSTRY_KEYWORDS[top2_name]), 3)
        if top2_raw > 0 else 0.0
    )

    shown     = matched_kws[top1_name][:5]
    remaining = top1_raw - len(shown)
    reasoning = (
        f"Matched {top1_raw}/{len(INDUSTRY_KEYWORDS[top1_name])} keywords: "
        + ", ".join(shown)
        + (f" ... +{remaining} more" if remaining > 0 else "")
    )

    return {
        "industry_1":        top1_name,
        "industry_1_score":  top1_score,
        "industry_2":        top2_name if top2_raw > 0 else None,
        "industry_2_score":  top2_score,
        "routing_confidence": top1_score,
        "confidence_band":   _confidence_band(top1_score),
        "reasoning":         reasoning,
        "router_path":       "keyword",
    }


# ── Path B: vLM routing (Phase 2.2) ─────────────────────────────────────────

def vlm_route(page_image: Image.Image) -> dict:
    """
    Route using a small vision-language model for image uploads and scanned pages.
    Returns the same dict structure as keyword_route.

    Phase 2.2 implementation: integrate PaliGemma-3B or InternVL2-2B here.
    Current stub: returns None for industry_1 to signal unresolved routing.
    Caller falls back to classifying against all hypotheses when this returns None.
    """
    return {
        "industry_1":        None,
        "industry_1_score":  0.0,
        "industry_2":        None,
        "industry_2_score":  0.0,
        "routing_confidence": 0.0,
        "confidence_band":   "LOW",
        "reasoning":         "vLM router not yet implemented (Phase 2.2). Falling back to full hypothesis set.",
        "router_path":       "vlm",
    }
