import json
from pathlib import Path, PurePosixPath


class KnowledgeScopeConfigError(ValueError):
    pass


def load_knowledge_scopes(path: str | Path) -> dict[str, str]:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeScopeConfigError(
            f"Could not read knowledge scope catalog {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise KnowledgeScopeConfigError(
            "Knowledge scope catalog must be a JSON object."
        )

    scopes = {}
    for raw_name, raw_prefix in payload.items():
        if not isinstance(raw_name, str) or not isinstance(raw_prefix, str):
            raise KnowledgeScopeConfigError(
                "Knowledge scope names and folder paths must be strings."
            )
        name = raw_name.strip().casefold()
        prefix = raw_prefix.replace("\\", "/").strip().rstrip("/")
        parts = PurePosixPath(prefix).parts
        if not name:
            raise KnowledgeScopeConfigError(
                "Knowledge scope names must not be empty."
            )
        if name == "all":
            raise KnowledgeScopeConfigError(
                "Knowledge scope name 'all' is reserved for unrestricted search."
            )
        if (
            not parts
            or parts[0] != "documents"
            or "." in parts
            or ".." in parts
            or PurePosixPath(prefix).is_absolute()
        ):
            raise KnowledgeScopeConfigError(
                f"Knowledge scope {raw_name!r} must point inside documents/."
            )
        if name in scopes:
            raise KnowledgeScopeConfigError(
                f"Duplicate knowledge scope name after normalization: {name!r}."
            )
        scopes[name] = PurePosixPath(*parts).as_posix()
    return scopes


def source_is_in_scope(source: str, prefix: str) -> bool:
    source_parts = PurePosixPath(source.replace("\\", "/")).parts
    prefix_parts = PurePosixPath(prefix).parts
    return source_parts[: len(prefix_parts)] == prefix_parts
