import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import classifier as clf
import postprocessor as post
import preprocessor as pre
from classifier import CATEGORIES_PATH, start_watcher

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
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


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


# ── Static files ──────────────────────────────────────────────────────────────
# Mounted last so all API routes above take precedence over file lookups.
# Serves CSS, JS, images and any other static assets from frontend/.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
