"""Expose Task 6 generation pipeline components."""

from pipelines.generation.citations import (
    Citation,
    CitationTracker,
    ParsedCitations,
)
from pipelines.generation.generator import (
    GenerationPipeline,
    GenerationResult,
)
from pipelines.generation.prompt_builder import (
    AssembledPrompt,
    PromptBuilder,
    SourceMapEntry,
)

__all__ = [
    "PromptBuilder",
    "AssembledPrompt",
    "SourceMapEntry",
    "CitationTracker",
    "Citation",
    "ParsedCitations",
    "GenerationPipeline",
    "GenerationResult",
]
