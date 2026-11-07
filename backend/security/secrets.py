"""Secret detection and redaction for untrusted text."""

import re

SECRET_PATTERNS = [
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("generic_secret", re.compile(r"['\"]?(?:api|secret|token|key|password)['\"]?\s*[:=]\s*['\"][A-Za-z0-9/+]{20,}['\"]", re.IGNORECASE)),
    ("private_key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----", re.DOTALL)),
    ("jwt", re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])")),
]


def redact_metadata(value):
    """Redact string values recursively without recording their contents."""
    if isinstance(value, str):
        from backend.security.validators import sanitize_text
        return redact_secrets(sanitize_text(value))[0]
    if isinstance(value, list):
        return [redact_metadata(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_metadata(item) for key, item in value.items()}
    return value


def redact_secrets(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for name, pattern in SECRET_PATTERNS:
        text, count = pattern.subn("[REDACTED]", text)
        if count:
            found.extend([name] * count)
    return text, found
