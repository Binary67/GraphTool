# Changelog

This changelog records changes from July 24, 2026 onward. Each entry explains
the intention behind the work and the features or code changes that implemented
it.

## July 24, 2026

### Intention

- Let users limit a question to a known document folder using natural language,
  without requiring them to know or enter an internal source path.
- Preserve broad knowledge-base search as the default when the user does not
  request a folder.
- Prevent an unknown or ambiguous folder request from silently searching the
  complete knowledge base.
- Keep folder-scoped answers grounded only in document chunks and graph
  relationships supported by the selected folder.

### Changes implemented

- Added `config/knowledge_scopes.json` as the catalog that maps user-facing scope
  names such as `work`, `personal`, and `finance` to folders under `documents/`.
- Added catalog loading and validation, including normalized scope names, safe
  document-folder paths, folder-boundary matching, and a reserved `all` name for
  unrestricted search.
- Extended question decomposition to select one catalog scope only when the user
  explicitly requests a folder. Questions without a folder restriction search
  all documents.
- Added clarification behavior for folder requests that do not match the
  catalog.
- Stored the selected scope once for the question and automatically applied it
  to every knowledge search performed for its subquestions.
- Added scoped runtime retrieval that filters both document chunks and
  source-provenance graph entities and relationships.
- Added per-scope retrieval caching while retaining the existing unrestricted
  search index.
- Documented the scope catalog and natural-language behavior in `README.md`.
- Added focused tests for catalog validation, default unrestricted search,
  selected-scope enforcement, unknown-scope clarification, source filtering, and
  graph-path filtering.
