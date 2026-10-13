"""Turn text / PDF / image uploads into plain text (OCR fallback for scans)."""
from __future__ import annotations
import io
from pathlib import Path


def _ocr_image(img) -> str:
    import pytesseract
    return pytesseract.image_to_string(img)


def read_document(name: str, data: bytes) -> tuple[str, str]:
    """Return (text, method). method describes how the text was obtained."""
    ext = Path(name).suffix.lower()
    if ext in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace"), "plain text"
    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
            if len(text.strip()) >= 40:
                return text, "PDF text layer"
            # scanned PDF -> rasterise and OCR
            parts = [_ocr_image(p.to_image(resolution=200).original) for p in pdf.pages]
            return "\n".join(parts), "PDF scan (OCR)"
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        from PIL import Image
        return _ocr_image(Image.open(io.BytesIO(data))), "image (OCR)"
    raise ValueError(f"Unsupported file type: {ext}")
