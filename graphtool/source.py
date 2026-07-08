import hashlib
import re
from pathlib import PurePosixPath


def source_key(source: str) -> str:
    normalized = source.replace("\\", "/").strip()
    stem = PurePosixPath(normalized).stem
    slug = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower() or "source"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"
