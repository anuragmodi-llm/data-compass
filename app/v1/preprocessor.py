import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pdf2image import convert_from_bytes, convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
from transformers import AutoProcessor

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff"}
PROCESSOR_MODEL = "google/siglip2-base-patch16-224"

_processor: AutoProcessor | None = None


def get_processor() -> AutoProcessor:
    global _processor
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(PROCESSOR_MODEL)
    return _processor


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        mask = image.split()[-1]
        background.paste(image.convert("RGB"), mask=mask)
        return background
    return image.convert("RGB")


def _images_from_pdf(source: bytes | str | Path, dpi: int = 200) -> list[Image.Image]:
    try:
        if isinstance(source, bytes):
            pages = convert_from_bytes(source, dpi=dpi)
        else:
            pages = convert_from_path(str(source), dpi=dpi)
    except PDFPageCountError as exc:
        raise ValueError(f"Could not determine page count — PDF may be corrupt or encrypted: {exc}") from exc
    except PDFSyntaxError as exc:
        raise ValueError(f"PDF syntax error — file may be corrupt: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to convert PDF to images: {exc}") from exc

    if not pages:
        raise ValueError("PDF produced no pages after conversion.")

    return [_to_rgb(p) for p in pages]


def _images_from_image(source: bytes | str | Path) -> list[Image.Image]:
    try:
        if isinstance(source, bytes):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(str(source))
        img.load()
    except UnidentifiedImageError as exc:
        raise ValueError(f"File is not a recognised image format: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to open image file: {exc}") from exc

    return [_to_rgb(img)]


def preprocess(source: bytes | str | Path) -> dict:
    """
    Accept a file path or raw bytes for a PDF or image.

    Returns:
        {
            "images":     [PIL.Image, ...],   # one entry per page / frame
            "page_count": int,
            "processor":  AutoProcessor,      # loaded from PROCESSOR_MODEL
        }

    Raises:
        ValueError  for corrupt, empty, or unsupported files.
        TypeError   if source is not bytes, str, or Path.
    """
    if not isinstance(source, (bytes, str, Path)):
        raise TypeError(f"source must be bytes, str, or Path — got {type(source).__name__}")

    # Determine format from extension when source is a path
    if isinstance(source, (str, Path)):
        ext = Path(source).suffix.lower()
        if ext == ".pdf":
            images = _images_from_pdf(source)
        elif ext in SUPPORTED_IMAGE_EXTS:
            images = _images_from_image(source)
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported types: .pdf, {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}"
            )
    else:
        # bytes: sniff format via Pillow; fall back to PDF on failure
        try:
            probe = Image.open(io.BytesIO(source))
            fmt = (probe.format or "").lower()
        except UnidentifiedImageError:
            fmt = "pdf"

        if fmt == "pdf" or fmt == "":
            images = _images_from_pdf(source)
        elif fmt in ("png", "jpeg", "tiff"):
            images = _images_from_image(source)
        else:
            raise ValueError(f"Unsupported image format '{fmt}' detected in bytes payload.")

    return {
        "images": images,
        "page_count": len(images),
        "processor": get_processor(),
    }
