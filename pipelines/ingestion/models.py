"""Shared data structures for multi-modal document extraction."""

from dataclasses import dataclass, field
from typing import Any, Literal


ContentType = Literal["text", "table", "image", "metadata"]


@dataclass
class ExtractedPage:
    """One extracted page or logical block from a source document."""

    page_number: int
    content: str
    content_type: ContentType = "text"
    metadata: dict[str, Any] = field(default_factory=dict)
