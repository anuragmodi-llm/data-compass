"""
Page splitting module.
Converts an uploaded PDF into rendered page images.
Returns the open fitz.Document — caller must call doc.close() after use.
"""

import fitz  # pymupdf
from PIL import Image


def split_pdf(file_bytes: bytes) -> tuple[list[Image.Image], fitz.Document]:
    """
    Open a PDF from raw bytes and render each page as a PIL Image at 150 DPI.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        page_images: List of PIL Images, one per page, rendered at 150 DPI.
                     Index 0 = page 1, index 1 = page 2, etc. (0-based internally).
        doc: The open fitz.Document. Caller MUST call doc.close() in a
             finally block after all pipeline steps are complete.

    Raises:
        ValueError: If the bytes cannot be parsed as a PDF.
        ValueError: If the PDF contains zero pages.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not open file as PDF: {e}")

    if len(doc) == 0:
        doc.close()
        raise ValueError("PDF contains zero pages.")

    page_images: list[Image.Image] = []
    mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images.append(img)

    return page_images, doc


def open_image(file_bytes: bytes) -> list[Image.Image]:
    """
    Open a single image (JPEG/PNG) from raw bytes as a one-item list.
    Used for image uploads which bypass the PDF splitting path.

    Args:
        file_bytes: Raw bytes of the uploaded image file.

    Returns:
        List containing one PIL Image.
    """
    import io
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB")
    return [img]
