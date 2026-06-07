"""
V2 Supabase logging — one row per detected document segment.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client

# app/v2/db.py is three levels deep → go up three parents to reach project root .env
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def log_to_supabase_v2(
    pdf_id: str,
    filename: str,
    results: list[dict],
    boundaries_suppressed: int = 0,
) -> None:
    """
    Insert one row into uploads_v2 for each detected document in results.
    Exceptions are caught and logged — classification continues even if logging fails.
    """
    try:
        client = _get_client()
        rows = []
        for result in results:
            b = result["boundary"]
            r = result["routing"]
            c = result["classification"]
            rows.append({
                "pdf_id":                         pdf_id,
                "filename":                       filename,
                "document_index":                 result["document_index"],
                "page_start":                     result["page_range"][0],
                "page_end":                       result["page_range"][1],
                "boundary_confidence":            b["confidence"],
                "boundary_confidence_band":       b["confidence_band"],
                "boundary_reasoning":             b["reasoning"],
                "boundaries_suppressed":          boundaries_suppressed,
                "industry_1":                     r["industry_1"],
                "industry_1_score":               r["industry_1_score"],
                "industry_2":                     r["industry_2"],
                "industry_2_score":               r["industry_2_score"],
                "routing_confidence":             r["confidence"],
                "routing_confidence_band":        r["confidence_band"],
                "routing_reasoning":              r["reasoning"],
                "router_path":                    r["router_path"],
                "predicted_class":                c["predicted_class"],
                "classification_confidence":      c["confidence"],
                "classification_confidence_band": c["confidence_band"],
                "classification_reasoning":       c["reasoning"],
                "top_hypothesis":                 c["top_hypothesis"],
            })
        client.table("uploads_v2").insert(rows).execute()
    except Exception:
        logger.exception("Failed to log v2 results to Supabase (pdf_id=%s)", pdf_id)


def fetch_v2_metrics() -> dict:
    """
    Call the get_v2_metrics() RPC function and return its JSON result.
    Returns {"status": "no_data"} if the table is empty or the call fails.
    """
    try:
        client = _get_client()
        result = client.rpc("get_v2_metrics", {}).execute()
        data = result.data
        if not data:
            return {"status": "no_data"}
        return data
    except Exception:
        logger.exception("Failed to fetch v2 metrics from Supabase")
        return {"status": "no_data"}
