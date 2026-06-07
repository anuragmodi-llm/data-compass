import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.v1.db import (
    log_upload,
    upload_file_to_storage,
    fetch_logs,
    fetch_class_distribution,
    fetch_confidence_breakdown,
    fetch_volume_by_day,
    fetch_summary_stats,
    get_signed_url,
)

import app.v1.classifier as clf
import app.v1.postprocessor as post
import app.v1.preprocessor as pre
from app.v1.classifier import CATEGORIES_PATH, start_watcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
}

_watcher = None

# Resolve frontend/ relative to the project root (one level above app/)
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _watcher
    logger.info("Preloading model and category tree …")
    clf._ensure_model()
    clf.get_category_tree()
    _watcher = start_watcher()
    logger.info("Startup complete.")
    yield
    if _watcher:
        _watcher.stop()
        _watcher.join()
    logger.info("Shutdown complete.")


app = FastAPI(title="Document Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error(message: str, code: int) -> JSONResponse:
    return JSONResponse(status_code=code, content={"error": message})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/classify")
async def classify_document(file: UploadFile = File(...)):
    # Content-type guard (best-effort; clients can lie, so we also catch errors below)
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        return _error(
            f"Unsupported file type '{file.content_type}'. "
            f"Accepted: PDF, PNG, JPEG, TIFF.",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    raw = await file.read()

    if len(raw) == 0:
        return _error("Uploaded file is empty.", status.HTTP_400_BAD_REQUEST)

    if len(raw) > MAX_FILE_BYTES:
        return _error(
            f"File size {len(raw) / 1_048_576:.1f} MB exceeds the 20 MB limit.",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    filename = file.filename or ""
    source = raw if not filename else filename  # prefer path sniffing when name is available
    start_time = time.time()

    try:
        preprocessed = pre.preprocess(raw)
    except (ValueError, TypeError) as exc:
        return _error(f"Preprocessing failed: {exc}", status.HTTP_422_UNPROCESSABLE_ENTITY)
    except Exception as exc:
        logger.exception("Unexpected preprocessing error")
        return _error(f"Preprocessing error: {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR)

    try:
        result = clf.classify(preprocessed)
    except Exception as exc:
        logger.exception("Classification error")
        return _error(f"Classification failed: {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR)

    tree = clf.get_category_tree()
    response = post.build_response(result, tree)

    # --- Logging hook ---
    try:
        upload_id = str(uuid.uuid4())
        processing_ms = int((time.time() - start_time) * 1000)
        storage_path = upload_file_to_storage(
            file_bytes=raw,
            filename=file.filename,
            upload_id=upload_id,
        )
        log_upload(
            upload_id=upload_id,
            filename=file.filename,
            file_type=file.content_type,
            file_size_bytes=len(raw),
            storage_path=storage_path,
            predicted_label=response.get("predicted_label"),
            confidence_score=response.get("confidence"),   # percentage 0-100, matches postprocessor output
            confidence_band=response.get("confidence_band"),
            processing_time_ms=processing_ms,
        )
    except Exception as exc:
        logger.warning("Logging/storage failed (non-fatal): %s", exc)
    # --- End logging hook ---

    return JSONResponse(content=response)


@app.get("/health")
def health():
    tree = clf.get_category_tree()
    model_loaded = clf._model is not None

    try:
        mtime = CATEGORIES_PATH.stat().st_mtime
        yaml_last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        yaml_last_modified = None

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": str(clf._device),
        "categories_count": len(tree.categories),
        "yaml_last_modified": yaml_last_modified,
    }


@app.get("/categories")
def list_categories():
    tree = clf.get_category_tree()
    return {
        "categories": [
            {"label": cat.label, "display_name": cat.display_name}
            for cat in tree.categories
        ]
    }


# ── Analytics & history endpoints ────────────────────────────────────────────

@app.get("/logs")
async def get_logs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    days: int = Query(default=30, ge=1, le=365),
):
    """Paginated upload history."""
    return fetch_logs(page=page, page_size=page_size, days=days)


@app.get("/stats/class-distribution")
async def get_class_distribution(days: int = Query(default=30, ge=1, le=365)):
    """Count per predicted label."""
    return fetch_class_distribution(days=days)


@app.get("/stats/confidence-breakdown")
async def get_confidence_breakdown(days: int = Query(default=30, ge=1, le=365)):
    """Count per confidence band."""
    return fetch_confidence_breakdown(days=days)


@app.get("/stats/volume")
async def get_volume(days: int = Query(default=30, ge=1, le=365)):
    """Upload count grouped by day."""
    return fetch_volume_by_day(days=days)


@app.get("/stats/summary")
async def get_summary(days: int = Query(default=30, ge=1, le=365)):
    """Scalar summary stats for the header cards."""
    return fetch_summary_stats(days=days)


@app.get("/file-url/{upload_id}")
async def get_file_url(upload_id: str):
    """Returns a 5-minute signed URL to view the original uploaded file."""
    from app.v1.db import get_client
    client = get_client()
    row = (
        client.table("uploads")
        .select("storage_path")
        .eq("id", upload_id)
        .single()
        .execute()
    )
    if not row.data:
        return JSONResponse(status_code=404, content={"error": "Upload not found"})
    signed_url = get_signed_url(row.data["storage_path"])
    return {"url": signed_url}


# ── Static files ──────────────────────────────────────────────────────────────
# Mounted last so all API routes above take precedence over file lookups.
# Serves CSS, JS, images and any other static assets from frontend/.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
