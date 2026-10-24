
"""Extract article text and metadata from web pages asynchronously."""

import asyncio
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from pipelines.ingestion.models import ExtractedPage


USER_AGENT = "NeuroFlowBot/1.0"


async def _is_allowed(url: str, client: httpx.AsyncClient) -> bool:
    """Check whether robots.txt permits fetching the URL."""
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False

    robots_url = urljoin(
        f"{parsed.scheme}://{parsed.netloc}",
        "/robots.txt",
    )

    try:
        response = await client.get(robots_url)
        if response.status_code == 404:
            return True
        response.raise_for_status()
    except httpx.HTTPError:
        # Fail closed if robots.txt cannot be checked.
        return False

    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())
    return parser.can_fetch(USER_AGENT, url)


async def extract_url(url: str) -> list[ExtractedPage]:
    """Fetch a URL and extract its main content and metadata."""
    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        if not await _is_allowed(url, client):
            raise ValueError("URL is invalid or disallowed by robots.txt.")

        response = await client.get(url)
        response.raise_for_status()

        final_url = str(response.url)
        html = response.text

    extracted = await asyncio.to_thread(
        trafilatura.extract,
        html,
        url=final_url,
        include_tables=True,
        output_format="txt",
        with_metadata=True,
        favor_precision=True,
    )

    if not extracted:
        raise ValueError("No extractable article content found at this URL.")

    metadata = await asyncio.to_thread(
        trafilatura.extract_metadata,
        html,
        default_url=final_url,
    )

    title = metadata.title if metadata else None
    author = metadata.author if metadata else None
    date = metadata.date if metadata else None
    canonical_url = metadata.url if metadata and metadata.url else final_url

    return [
        ExtractedPage(
            page_number=1,
            content=extracted,
            content_type="text",
            metadata={
                "source_type": "url",
                "title": title,
                "author": author,
                "canonical_url": canonical_url,
                "publish_date": date,
                "fetched_url": final_url,
            },
        )
    ]
