"""
db.py — Supabase client + insert/query functions
Place this file at: app/db.py
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# .env lives at the project root (one level above this file's app/ directory)
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
BUCKET = "documents"

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------

def upload_file_to_storage(file_bytes: bytes, filename: str, upload_id: str) -> str:
    """
    Uploads file bytes to Supabase Storage.
    Returns the storage path: {upload_id}/{filename}
    """
    path = f"{upload_id}/{filename}"
    client = get_client()
    client.storage.from_(BUCKET).upload(
        path=path,
        file=file_bytes,
        file_options={"content-type": "application/octet-stream"},
    )
    return path


def get_signed_url(storage_path: str, expires_in: int = 300) -> str:
    """
    Returns a signed URL valid for `expires_in` seconds (default 5 min).
    """
    client = get_client()
    result = client.storage.from_(BUCKET).create_signed_url(storage_path, expires_in)
    return result["signedUrl"]


# ------------------------------------------------------------------
# Database
# ------------------------------------------------------------------

def log_upload(
    upload_id: str,
    filename: str,
    file_type: str,
    file_size_bytes: int,
    storage_path: str,
    predicted_label: str,
    confidence_score: float,
    confidence_band: str,
    processing_time_ms: int,
) -> dict:
    """
    Inserts one row into the uploads table.
    Returns the inserted row.
    """
    client = get_client()
    row = {
        "id": upload_id,
        "filename": filename,
        "file_type": file_type,
        "file_size_bytes": file_size_bytes,
        "storage_path": storage_path,
        "predicted_label": predicted_label,
        "confidence_score": confidence_score,
        "confidence_band": confidence_band,
        "processing_time_ms": processing_time_ms,
    }
    result = client.table("uploads").insert(row).execute()
    return result.data[0] if result.data else row


def fetch_logs(page: int = 1, page_size: int = 20, days: int = 30) -> dict:
    """
    Returns paginated upload history, newest first.
    Filters to the last `days` days.
    """
    client = get_client()
    offset = (page - 1) * page_size

    # Total count
    count_result = (
        client.table("uploads")
        .select("id", count="exact")
        .gte("uploaded_at", _days_ago(days))
        .execute()
    )
    total = count_result.count or 0

    # Page of rows
    rows = (
        client.table("uploads")
        .select("*")
        .gte("uploaded_at", _days_ago(days))
        .order("uploaded_at", desc=True)
        .range(offset, offset + page_size - 1)
        .execute()
    )

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": rows.data or [],
    }


def fetch_class_distribution(days: int = 30) -> list[dict]:
    """
    Returns count per predicted_label.
    """
    client = get_client()
    # Supabase doesn't support group-by natively via the client; use rpc or raw SQL
    result = client.rpc("class_distribution", {"days_back": days}).execute()
    return result.data or []


def fetch_confidence_breakdown(days: int = 30) -> list[dict]:
    """
    Returns count per confidence_band.
    """
    client = get_client()
    result = client.rpc("confidence_breakdown", {"days_back": days}).execute()
    return result.data or []


def fetch_volume_by_day(days: int = 30) -> list[dict]:
    """
    Returns upload count grouped by day for last `days` days.
    """
    client = get_client()
    result = client.rpc("volume_by_day", {"days_back": days}).execute()
    return result.data or []


def fetch_summary_stats(days: int = 30) -> dict:
    """
    Returns scalar summary stats: total, unique classes, rejection rate, avg confidence.
    """
    client = get_client()
    result = client.rpc("summary_stats", {"days_back": days}).execute()
    return result.data[0] if result.data else {}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _days_ago(days: int) -> str:
    from datetime import timedelta
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.isoformat()
