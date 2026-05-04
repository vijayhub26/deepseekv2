"""
PDF Processor — Converts multi-page PDFs into individual page images.

Uses PyMuPDF (fitz) to render each page at a configurable DPI,
returning PIL Image objects ready for OCR inference.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


def validate_pdf(pdf_path: str) -> Path:
    """Validate that the given path points to an existing PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")
    return path


def pdf_to_images(
    pdf_path: str,
    dpi: int = 300,
    output_dir: str = None,
) -> List[Tuple[int, Image.Image]]:
    """Convert each page of a PDF into a PIL Image.

    Args:
        pdf_path: Path to the input PDF file.
        dpi: Render resolution in dots per inch. Default 300.
        output_dir: If provided, saves each page image as a PNG to this
                     directory. Otherwise images are kept in memory only.

    Returns:
        A list of (page_number, PIL.Image) tuples. Page numbers are 1-indexed.
    """
    path = validate_pdf(pdf_path)
    zoom = dpi / 72  # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(str(path))
    total_pages = len(doc)
    logger.info(f"Opened PDF: {path.name} ({total_pages} pages, rendering at {dpi} DPI)")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    page_images: List[Tuple[int, Image.Image]] = []

    for page_idx in range(total_pages):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=matrix)

        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        page_num = page_idx + 1
        page_images.append((page_num, img))
        logger.debug(f"  Page {page_num}/{total_pages}: {pix.width}x{pix.height} px")

        # Optionally save to disk
        if output_dir:
            img_path = os.path.join(output_dir, f"page_{page_num:04d}.png")
            img.save(img_path, "PNG")
            logger.debug(f"  Saved: {img_path}")

    doc.close()
    logger.info(f"Extracted {total_pages} page images from {path.name}")
    return page_images


def get_pdf_info(pdf_path: str) -> dict:
    """Get basic metadata about a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary with keys: 'path', 'name', 'pages', 'metadata'.
    """
    path = validate_pdf(pdf_path)
    doc = fitz.open(str(path))
    info = {
        "path": str(path),
        "name": path.name,
        "pages": len(doc),
        "metadata": doc.metadata,
    }
    doc.close()
    return info
