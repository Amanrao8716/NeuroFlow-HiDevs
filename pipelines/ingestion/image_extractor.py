"""Image extraction using OCR and NeuroFlow's vision model router."""

import asyncio
import base64
from io import BytesIO

import pytesseract
from PIL import Image

from providers.base import ChatMessage
from providers.client import NeuroFlowClient
from providers.router import RoutingCriteria
from pipelines.ingestion.models import ExtractedPage


MAX_IMAGE_SIDE = 1024


def _prepare_image(image_bytes: bytes) -> tuple[Image.Image, str]:
    """Resize an image and return it with its MIME type."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return image, base64.b64encode(buffer.getvalue()).decode("ascii")


async def extract_image(
    image_bytes: bytes,
    client: NeuroFlowClient,
) -> list[ExtractedPage]:
    """Combine a vision description with OCR text."""
    image, image_base64 = _prepare_image(image_bytes)

    try:
        try:
            ocr_text = await asyncio.to_thread(
                pytesseract.image_to_string,
                image,
                config="--psm 6",
            )
            ocr_text = ocr_text.strip()
            ocr_status = "completed"
        except pytesseract.TesseractNotFoundError:
            ocr_text = ""
            ocr_status = "unavailable"

        description = ""
        vision_status = "completed"

        try:
            result = await client.chat(
                [
                    ChatMessage(
                        role="user",
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    "Describe this image accurately. "
                                    "Include its important objects, "
                                    "layout, and visible information."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/jpeg;base64,"
                                        + image_base64
                                    )
                                },
                            },
                        ],
                    )
                ],
                routing_criteria=RoutingCriteria(
                    task_type="image_description",
                    require_vision=True,
                ),
            )
            description = result.content.strip()
        except Exception:
            # Preserve OCR output if vision routing or generation fails.
            vision_status = "unavailable"

        content = description
        if ocr_text:
            content += "\n\nText found in image: " + ocr_text

        if not content:
            content = "No text or image description could be extracted."

        return [
            ExtractedPage(
                page_number=1,
                content=content,
                content_type="image",
                metadata={
                    "source_type": "image",
                    "ocr_status": ocr_status,
                    "vision_status": vision_status,
                    "image_width": image.width,
                    "image_height": image.height,
                },
            )
        ]
    finally:
        image.close()
