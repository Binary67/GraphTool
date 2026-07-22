from graphtool import source
from graphtool.source import document_content_hash


def test_document_content_hash_uses_exact_text():
    assert document_content_hash("# Title\nText") == document_content_hash(
        "# Title\nText"
    )
    assert document_content_hash("# Title\nText") != document_content_hash(
        "# Title\nText\n"
    )


def test_document_content_hash_changes_with_ingestion_version(monkeypatch):
    content = "# Title\nText"
    original_hash = document_content_hash(content)

    monkeypatch.setattr(source, "INGESTION_VERSION", source.INGESTION_VERSION + 1)

    assert document_content_hash(content) != original_hash
