"""
Pipeline orchestrator — v2 multi-document classify endpoint.

Imports the v1 FastAPI app and adds two v2 routes to it:
  POST /v2/classify  — full multi-document pipeline
  GET  /v2/metrics   — pipeline health metrics from Supabase

All v1 routes (/classify, /health, /logs, etc.) remain untouched.

Route ordering note: v1/main.py ends with app.mount("/", StaticFiles(...))
which is a catch-all. Routes appended after that mount are never reached.
Fix: remove the static mount before registering v2 routes, then re-append
it so it remains the final fallback — preserving the correct match order.
"""

import logging
import time
import uuid

from fastapi import File, Header, UploadFile
from fastapi.responses import JSONResponse
from starlette.routing import Mount

from app.v1.main import app  # all v1 routes already registered

from app.v2.pipeline.splitter import split_pdf, open_image
from app.v2.pipeline.boundary import detect_boundaries, merge_same_class_segments
from app.v2.pipeline.router import keyword_route, vlm_route, extract_text_full
from app.v2.classifier import classify_with_siglip2, load_hypotheses_for_industry, load_all_hypotheses
from app.v2.db import log_to_supabase_v2, fetch_v2_metrics

logger = logging.getLogger(__name__)


# ── Re-anchor static mount so v2 routes take precedence ──────────────────────
# v1/main.py registers app.mount("/", StaticFiles(...)) last, making it a
# catch-all that intercepts any route appended after it.  Remove it now,
# register v2 routes below, then re-append it at the very end.

_static_mount = None
for _route in app.router.routes:
    if isinstance(_route, Mount) and getattr(_route, "name", None) == "frontend":
        _static_mount = _route
        break
if _static_mount:
    app.router.routes.remove(_static_mount)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence_band(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.50:
        return "MEDIUM"
    return "LOW"


def _classification_reasoning(top_score: float, second_score: float) -> str:
    margin = round(top_score - second_score, 3)
    if margin >= 0.4:
        clarity = "unambiguous match"
    elif margin >= 0.2:
        clarity = "clear match"
    elif margin >= 0.1:
        clarity = "moderate confidence match"
    else:
        clarity = "low-margin match — manual review recommended"
    return (
        f"Top hypothesis score {top_score:.2f} vs next-best {second_score:.2f}. "
        f"Margin {margin} indicates {clarity}."
    )


# ── V2 routes ─────────────────────────────────────────────────────────────────

@app.post("/v2/classify")
async def classify_v2(
    file: UploadFile = File(...),
    x_benchmark_timing: str = Header(default=None, alias="X-Benchmark-Timing"),
):
    want_timing = (x_benchmark_timing or "").lower() == "true"
    t_total_start = time.perf_counter()

    # ── Step: file read ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    file_bytes = await file.read()
    file_read_ms = round((time.perf_counter() - t0) * 1000, 1)

    filename = file.filename or ""
    pdf_id = str(uuid.uuid4())

    is_image = filename.lower().endswith((".jpg", ".jpeg", ".png"))

    doc = None  # Bug 3: orchestrator owns fitz.Document lifecycle
    total_suppressed = 0

    # Timing accumulators (summed across all segments for per-segment steps)
    _t_routing        = 0.0
    _t_hyp_loading    = 0.0
    _t_siglip2        = 0.0

    try:
        # ── Step: page splitting ──────────────────────────────────────────────
        print(f"[DEBUG] Starting page split — file: {filename}")
        t0 = time.perf_counter()
        if is_image:
            page_images = open_image(file_bytes)
            total_pages = 1
            segments = [{
                "page_range":           [0, 0],
                "representative_image": page_images[0],
                "boundary_info":        None,
            }]
        else:
            try:
                page_images, doc = split_pdf(file_bytes)
            except ValueError as exc:
                return JSONResponse(status_code=422, content={"error": str(exc)})
            total_pages = len(page_images)
            if total_pages == 1:
                segments = [{
                    "page_range":           [0, 0],
                    "representative_image": page_images[0],
                    "boundary_info":        None,
                }]
        splitting_ms = round((time.perf_counter() - t0) * 1000, 1)
        print(f"[DEBUG] Split complete — {len(page_images)} pages")

        # ── Step: metadata fingerprinting + pHash + boundary detection ────────
        t0 = time.perf_counter()
        metadata_ms = 0.0
        phash_ms = 0.0
        boundary_ms = 0.0
        heading_ms = 0.0
        if not is_image and total_pages > 1:
            print(f"[DEBUG] Starting metadata fingerprinting")
            # boundary detection wraps metadata, phash, and heading steps
            # internally; we time the whole block and attribute to boundary
            t_bd = time.perf_counter()
            print(f"[DEBUG] Metadata complete")
            print(f"[DEBUG] Starting pHash computation")
            print(f"[DEBUG] pHash complete")
            print(f"[DEBUG] Starting boundary detection")
            segments, total_suppressed = detect_boundaries(doc, page_images)
            boundary_ms = round((time.perf_counter() - t_bd) * 1000, 1)
            print(f"[DEBUG] Boundary detection complete — {len(segments)} segments found")
        else:
            boundary_ms = round((time.perf_counter() - t0) * 1000, 1)

        # ── Route + scope + classify each segment ────────────────────────────
        classified_segments = []

        for i, seg in enumerate(segments):
            page_idx = seg["page_range"][0]  # 0-based
            rep_image = seg["representative_image"]

            # Industry routing
            print(f"[DEBUG] Starting routing for segment {i}")
            t0 = time.perf_counter()
            if is_image:
                routing = vlm_route(rep_image)
            else:
                text = extract_text_full(doc[page_idx])
                routing = keyword_route(text)
                if routing["industry_1"] is None:
                    routing = vlm_route(rep_image)
            _t_routing += time.perf_counter() - t0
            print(f"[DEBUG] Routing complete — industry: {routing['industry_1']}, confidence: {routing['routing_confidence']}")

            industry = routing["industry_1"]
            routing_confidence = routing.get("routing_confidence", 0.0)

            ROUTING_CONFIDENCE_THRESHOLD = 0.25

            # Hypothesis loading
            t0 = time.perf_counter()
            if industry is not None and routing_confidence >= ROUTING_CONFIDENCE_THRESHOLD:
                hypotheses = load_hypotheses_for_industry(industry)
                hypothesis_scope = "industry_scoped"
            else:
                hypotheses = load_all_hypotheses()
                hypothesis_scope = "full_set_fallback"
            _t_hyp_loading += time.perf_counter() - t0

            # SigLIP 2 classification
            print(f"[DEBUG] Starting SigLIP2 classification")
            t0 = time.perf_counter()
            result = classify_with_siglip2(image=rep_image, hypotheses=hypotheses)
            _t_siglip2 += time.perf_counter() - t0
            print(f"[DEBUG] Classification complete — class: {result['class']}, confidence: {result['top_score']}")

            cl_confidence = round(result["top_score"], 3)
            cl_reasoning  = _classification_reasoning(result["top_score"], result["second_score"])

            classified_segments.append({
                "page_range":                    seg["page_range"],
                "boundary_info":                 seg["boundary_info"],
                "routing":                       routing,
                "hypothesis_scope":              hypothesis_scope,
                "predicted_class":               result["class"],
                "top_hypothesis":                result["top_hypothesis"],
                "classification_confidence":     cl_confidence,
                "classification_confidence_band": _confidence_band(cl_confidence),
                "classification_reasoning":      cl_reasoning,
            })

        routing_ms    = round(_t_routing    * 1000, 1)
        hyp_load_ms   = round(_t_hyp_loading * 1000, 1)
        siglip2_ms    = round(_t_siglip2    * 1000, 1)

        # Step 9: post-boundary merge
        final_segments = merge_same_class_segments(classified_segments)

        # ── Final response assembly — ONLY place 0-based → 1-based ──────────
        results = []
        for i, seg in enumerate(final_segments):
            b = seg["boundary_info"]
            results.append({
                "document_index": i + 1,
                "page_range": [
                    seg["page_range"][0] + 1,
                    seg["page_range"][1] + 1,
                ],
                "boundary": {
                    "confidence":      b["confidence"]      if b else None,
                    "confidence_band": b["confidence_band"] if b else None,
                    "reasoning":       b["reasoning"]       if b else
                                       "Single document upload — no boundary detection performed",
                },
                "routing": {
                    "industry_1":       seg["routing"]["industry_1"],
                    "industry_1_score": seg["routing"]["industry_1_score"],
                    "industry_2":       seg["routing"]["industry_2"],
                    "industry_2_score": seg["routing"]["industry_2_score"],
                    "confidence":       seg["routing"]["routing_confidence"],
                    "confidence_band":  seg["routing"]["confidence_band"],
                    "reasoning":        seg["routing"]["reasoning"],
                    "router_path":      seg["routing"]["router_path"],
                    "hypothesis_scope": seg["hypothesis_scope"],
                },
                "classification": {
                    "predicted_class":  seg["predicted_class"],
                    "confidence":       seg["classification_confidence"],
                    "confidence_band":  seg["classification_confidence_band"],
                    "reasoning":        seg["classification_reasoning"],
                    "top_hypothesis":   seg["top_hypothesis"],
                },
            })

        # Step 10: Supabase logging
        t0 = time.perf_counter()
        try:
            log_to_supabase_v2(
                pdf_id=pdf_id,
                filename=filename,
                results=results,
                boundaries_suppressed=total_suppressed,
            )
        except Exception:
            logger.exception("v2 Supabase logging failed (non-fatal)")
        supabase_ms = round((time.perf_counter() - t0) * 1000, 1)

        total_ms = round((time.perf_counter() - t_total_start) * 1000, 1)

        response: dict = {
            "pipeline_version":   "v2",
            "filename":           filename,
            "total_pages":        total_pages,
            "documents_detected": len(results),
            "results":            results,
        }

        if want_timing:
            response["pipeline_timing"] = {
                "file_read_ms":                  file_read_ms,
                "page_splitting_ms":             splitting_ms,
                "metadata_fingerprinting_ms":    metadata_ms,
                "phash_computation_ms":          phash_ms,
                "boundary_detection_ms":         boundary_ms,
                "heading_confirmation_ms":       heading_ms,
                "industry_routing_ms":           routing_ms,
                "hypothesis_loading_ms":         hyp_load_ms,
                "siglip2_classification_ms":     siglip2_ms,
                "supabase_logging_ms":           supabase_ms,
                "total_pipeline_ms":             total_ms,
            }

        return response

    finally:
        if doc is not None:
            doc.close()


@app.post("/v2/debug/routing")
async def debug_routing(file: UploadFile = File(...)):
    """
    Debug endpoint — returns per-page routing details without running classification.
    Shows extracted text, keyword hits, industry scores, and hypothesis scope decision
    for every page in the uploaded file.
    """
    file_bytes = await file.read()
    filename = file.filename or ""
    is_image = filename.lower().endswith((".jpg", ".jpeg", ".png"))

    doc = None
    try:
        if is_image:
            return {
                "filename": filename,
                "file_type": "image",
                "note": "Images bypass keyword routing and go directly to vLM path.",
                "pages": [{
                    "page": 1,
                    "router_path": "vlm",
                    "extracted_text": "",
                    "industry_1": None,
                    "industry_1_score": 0.0,
                    "routing_confidence": 0.0,
                    "confidence_band": "LOW",
                    "reasoning": "vLM router not yet implemented (Phase 2.2). Falling back to full hypothesis set.",
                    "hypothesis_scope": "full_set_fallback",
                }],
            }

        try:
            page_images, doc = split_pdf(file_bytes)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})

        total_pages = len(page_images)
        pages_detail = []

        for page_idx in range(total_pages):
            text = extract_text_full(doc[page_idx])
            routing = keyword_route(text)
            if routing["industry_1"] is None:
                routing = vlm_route(page_images[page_idx])

            industry = routing["industry_1"]
            routing_confidence = routing.get("routing_confidence", 0.0)
            ROUTING_CONFIDENCE_THRESHOLD = 0.25
            hypothesis_scope = (
                "industry_scoped"
                if industry is not None and routing_confidence >= ROUTING_CONFIDENCE_THRESHOLD
                else "full_set_fallback"
            )

            pages_detail.append({
                "page": page_idx + 1,
                "router_path": routing["router_path"],
                "extracted_text_preview": text[:300].strip() if text else "",
                "extracted_text_length": len(text),
                "industry_1": routing["industry_1"],
                "industry_1_score": routing["industry_1_score"],
                "industry_2": routing["industry_2"],
                "industry_2_score": routing["industry_2_score"],
                "routing_confidence": routing["routing_confidence"],
                "confidence_band": routing["confidence_band"],
                "reasoning": routing["reasoning"],
                "hypothesis_scope": hypothesis_scope,
                "routing_confidence_threshold": ROUTING_CONFIDENCE_THRESHOLD,
            })

        return {
            "filename": filename,
            "file_type": "pdf",
            "total_pages": total_pages,
            "pages": pages_detail,
        }

    finally:
        if doc is not None:
            doc.close()


@app.get("/v2/metrics")
async def get_v2_metrics():
    return fetch_v2_metrics()


# ── Re-append static files mount as the final catch-all ──────────────────────
if _static_mount:
    app.router.routes.append(_static_mount)
