from graphtool.source import document_content_hash


def test_document_content_hash_uses_exact_text():
    assert document_content_hash("# Title\nText") == document_content_hash(
        "# Title\nText"
    )
    assert document_content_hash("# Title\nText") != document_content_hash(
        "# Title\nText\n"
    )
