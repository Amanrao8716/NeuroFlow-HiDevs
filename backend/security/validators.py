"""Reusable text and public URL validation helpers."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit


def sanitize_text(value: str) -> str:
    # bleach is an explicit dependency. This small fallback only exists so
    # read-only tools can import the module before dependencies are installed.
    try:
        import bleach
    except ImportError as exc:
        raise RuntimeError("Install backend requirements to sanitize text safely.") from exc
    return bleach.clean(value, tags=[], strip=True)


def validate_public_url(url: str, *, resolve: bool = True) -> list[str]:
    """Validate an HTTP(S) URL and ensure every current DNS answer is public."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only valid HTTP(S) URLs are accepted.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Credentials in URLs are not allowed.")
        port = parsed.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("Invalid URL port.")
        expected_port = 443 if parsed.scheme.lower() == "https" else 80
        if port is not None and port != expected_port:
            raise ValueError("Unexpected URL port.")
        host = parsed.hostname.rstrip(".").lower()
        if host == "localhost" or host.endswith(".localhost") or ("." not in host and not _is_ip(host)):
            raise ValueError("Internal hostnames are not allowed.")
        if _is_ip(host):
            addresses = [str(ipaddress.ip_address(host))]
        elif resolve:
            addresses = sorted({item[4][0] for item in socket.getaddrinfo(host, port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)})
        else:
            addresses = []
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
                ip = ip.ipv4_mapped
            if not ip.is_global:
                raise ValueError("URL resolves to a non-public destination.")
        return addresses
    except (ValueError, OSError, socket.gaierror) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("Only", "Credentials", "Invalid", "Unexpected", "Internal", "URL resolves")):
            raise
        raise ValueError("URL is invalid or does not resolve to a public destination.") from exc


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False
