
"""Extract article text and metadata from web pages asynchronously."""

import asyncio
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import httpcore
import trafilatura

from backend.security.validators import validate_public_url

from pipelines.ingestion.models import ExtractedPage


USER_AGENT = "NeuroFlowBot/1.0"


class _PinnedBackend(httpcore.AsyncNetworkBackend):
    """Resolve the validated hostname once and pin its public address for sockets."""

    def __init__(self, hostname: str, addresses: list[str]):
        self.hostname = hostname.lower()
        self.addresses = addresses
        self.backend = httpcore.AsyncIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        if host.lower() != self.hostname:
            return await self.backend.connect_tcp(host, port, timeout, local_address, socket_options)
        last_error = None
        for address in self.addresses:
            try:
                return await self.backend.connect_tcp(address, port, timeout, local_address, socket_options)
            except (OSError, httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        raise last_error or OSError("No validated public address is reachable.")

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return await self.backend.connect_unix_socket(path, timeout, socket_options)

    async def sleep(self, seconds):
        await self.backend.sleep(seconds)


def _safe_transport(url: str) -> httpx.AsyncHTTPTransport:
    from urllib.parse import urlsplit

    addresses = validate_public_url(url)
    hostname = urlsplit(url).hostname or ""
    transport = httpx.AsyncHTTPTransport(retries=0)
    # httpx does not expose httpcore's network backend in its public transport
    # constructor; pin its pool backend before any request is sent.
    transport._pool._network_backend = _PinnedBackend(hostname, addresses)
    return transport


async def _is_allowed(url: str, client: httpx.AsyncClient) -> bool:
    """Check whether robots.txt permits fetching the URL."""
    try:
        validate_public_url(url)
    except ValueError:
        return False
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
        transport=_safe_transport(url),
        timeout=20.0,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        if not await _is_allowed(url, client):
            raise ValueError("URL is invalid or disallowed by robots.txt.")

        response = await client.get(url)
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise ValueError("URL redirect did not include a target.")
            redirect_url = urljoin(url, location)
            async with httpx.AsyncClient(
                transport=_safe_transport(redirect_url), timeout=20.0,
                follow_redirects=False, headers={"User-Agent": USER_AGENT},
            ) as redirect_client:
                redirect_response = await redirect_client.get(redirect_url)
            if redirect_response.is_redirect:
                raise ValueError("Multiple URL redirects are not allowed.")
            response = redirect_response
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
