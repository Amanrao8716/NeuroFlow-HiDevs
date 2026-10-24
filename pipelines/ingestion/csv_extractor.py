"""CSV extraction in 100-row blocks with large-file summaries."""

from io import BytesIO

import pandas as pd

from pipelines.ingestion.models import ExtractedPage


BLOCK_SIZE = 100
SUMMARY_THRESHOLD = 1000


def extract_csv(file_bytes: bytes) -> list[ExtractedPage]:
    """Extract CSV data as Markdown blocks or a large-file summary."""
    dataframe = pd.read_csv(BytesIO(file_bytes))

    if dataframe.empty and len(dataframe.columns) == 0:
        return []

    results: list[ExtractedPage] = []
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    text_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns
    ]

    column_metadata = {
        "columns": dataframe.columns.tolist(),
        "numeric_columns": numeric_columns,
        "text_columns": text_columns,
        "total_rows": len(dataframe),
    }

    if len(dataframe) > SUMMARY_THRESHOLD:
        summary = dataframe.describe(include="all").fillna("")
        sample = dataframe.head(10).fillna("")

        content = (
            "CSV statistical summary:\n\n"
            + summary.to_markdown()
            + "\n\nSample rows:\n\n"
            + sample.to_markdown(index=False)
        )

        results.append(
            ExtractedPage(
                page_number=1,
                content=content,
                content_type="text",
                metadata={
                    "source_type": "csv",
                    **column_metadata,
                    "representation": "summary_and_sample",
                },
            )
        )

    else:
        for start in range(0, len(dataframe), BLOCK_SIZE):
            block = dataframe.iloc[start : start + BLOCK_SIZE]
            content = block.to_markdown(index=False)

            results.append(
                ExtractedPage(
                    page_number=(start // BLOCK_SIZE) + 1,
                    content=content,
                    content_type="table",
                    metadata={
                        "source_type": "csv",
                        **column_metadata,
                        "row_start": start,
                        "row_end": start + len(block) - 1,
                        "representation": "row_block",
                    },
                )
            )

    return results
