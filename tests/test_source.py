import hashlib

from graphtool.source import INGESTION_VERSION, document_content_hash


def test_document_content_hash_uses_exact_text():
    assert document_content_hash("# Title\nText") == document_content_hash(
        "# Title\nText"
    )
    assert document_content_hash("# Title\nText") != document_content_hash(
        "# Title\nText\n"
    )


def test_document_content_hash_includes_ingestion_version():
    content = "# Title\nText"
    payload = f"graphtool-ingestion-v{INGESTION_VERSION}\0{content}"

    assert document_content_hash(content) == hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
