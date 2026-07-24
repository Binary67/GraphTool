import pytest

from graphtool.knowledge_scopes import (
    KnowledgeScopeConfigError,
    load_knowledge_scopes,
    source_is_in_scope,
)


def test_load_knowledge_scopes_normalizes_names_and_paths(tmp_path):
    path = tmp_path / "knowledge_scopes.json"
    path.write_text(
        '{" Work ": "documents\\\\work\\\\", "personal": "documents/personal"}'
    )

    assert load_knowledge_scopes(path) == {
        "work": "documents/work",
        "personal": "documents/personal",
    }


def test_load_knowledge_scopes_returns_empty_for_missing_catalog(tmp_path):
    assert load_knowledge_scopes(tmp_path / "missing.json") == {}


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"work": "../work"}',
        '{"work": "/documents/work"}',
        '{"work": 1}',
        '{"all": "documents/archive"}',
    ],
)
def test_load_knowledge_scopes_rejects_invalid_catalog(tmp_path, payload):
    path = tmp_path / "knowledge_scopes.json"
    path.write_text(payload)

    with pytest.raises(KnowledgeScopeConfigError):
        load_knowledge_scopes(path)


def test_source_scope_matching_respects_folder_boundaries():
    assert source_is_in_scope(
        "documents/work/project/plan.md",
        "documents/work",
    )
    assert not source_is_in_scope(
        "documents/work-personal/plan.md",
        "documents/work",
    )
