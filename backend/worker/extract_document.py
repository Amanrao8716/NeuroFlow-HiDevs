"""Container entry point for one isolated document extraction."""

import argparse
import asyncio
import base64
import json
from pathlib import Path
from io import BytesIO

from pipelines.ingestion.csv_extractor import extract_csv
from pipelines.ingestion.docx_extractor import extract_docx
from pipelines.ingestion.image_extractor import extract_image
from pipelines.ingestion.models import ExtractedPage
from PIL import Image
import pytesseract
from pipelines.ingestion.pdf_extractor import extract_pdf

MAX_OUTPUT_BYTES = 50 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-type", choices=["pdf", "docx", "csv", "image"], required=True)
    args = parser.parse_args()
    data = Path("/input/source.bin").read_bytes()
    if args.source_type == "pdf":
        pages = extract_pdf(data)
    elif args.source_type == "docx":
        pages = extract_docx(data)
    elif args.source_type == "csv":
        pages = extract_csv(data)
    else:
        image = Image.open(BytesIO(data)).convert("RGB")
        image.thumbnail((1024, 1024))
        text = pytesseract.image_to_string(image, config="--psm 6").strip()
        pages = [ExtractedPage(page_number=1, content=text, content_type="image", metadata={"source_type": "image", "vision_status": "disabled_in_isolated_sandbox"})]
    encoded = json.dumps([{
            "page_number": page.page_number,
            "content": page.content,
            "content_type": page.content_type,
            "metadata": page.metadata,
        } for page in pages])
    if len(encoded.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise RuntimeError("Extracted output exceeds configured limit.")
    Path("/output/pages.json").write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
