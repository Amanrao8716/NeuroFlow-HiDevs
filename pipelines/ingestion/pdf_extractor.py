"""PDF extraction with OCR fallback and table extraction."""

from io import BytesIO

import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from PIL import Image

from pipelines.ingestion.models import ExtractedPage


OCR_THRESHOLD = 50
OCR_SCALE = 2


def _ocr_page(page) -> str:
    """Rasterize a PDF page and extract text using Tesseract."""
    bitmap = page.render(scale=OCR_SCALE)
    image = bitmap.to_pil()
    try:
        return pytesseract.image_to_string(
            image,
            config="--psm 6",
        ).strip()
    finally:
        image.close()


def extract_pdf(file_bytes: bytes) -> list[ExtractedPage]:
    """Extract PDF text and tables, using OCR for scanned pages."""
    results: list[ExtractedPage] = []
    pdf = pdfium.PdfDocument(file_bytes)

    try:
        with pdfplumber.open(BytesIO(file_bytes)) as table_pdf:
            for index, page in enumerate(pdf):
                page_number = index + 1
                text_page = page.get_textpage()

                try:
                    text = text_page.get_text_range().strip()
                finally:
                    text_page.close()

                needs_ocr = len(text) < OCR_THRESHOLD
                ocr_attempted = False

                if needs_ocr:
                    ocr_attempted = True
                    try:
                        ocr_text = _ocr_page(page)
                        if ocr_text:
                            text = ocr_text
                    except pytesseract.TesseractNotFoundError:
                        pass

                results.append(
                    ExtractedPage(
                        page_number=page_number,
                        content=text,
                        content_type="text",
                        metadata={
                            "source_type": "pdf",
                            "needs_ocr": needs_ocr,
                            "ocr_attempted": ocr_attempted,
                            "ocr_status": (
                                "completed"
                                if needs_ocr and text
                                else "unavailable"
                                if needs_ocr
                                else "not_required"
                            ),
                        },
                    )
                )

                if index < len(table_pdf.pages):
                    table_page = table_pdf.pages[index]

                    for table_index, table in enumerate(
                        table_page.extract_tables()
                    ):
                        rows = [
                            [
                                (cell or "").replace("\n", " ").strip()
                                for cell in row
                            ]
                            for row in table
                        ]

                        if not rows:
                            continue

                        markdown = "\n".join(
                            "| " + " | ".join(row) + " |"
                            for row in rows
                        )

                        results.append(
                            ExtractedPage(
                                page_number=page_number,
                                content=markdown,
                                content_type="table",
                                metadata={
                                    "source_type": "pdf",
                                    "table_index": table_index,
                                },
                            )
                        )
    finally:
        pdf.close()

    return results
