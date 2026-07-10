# Changelog

## [Unreleased]

### Added

- Added exact document content hashing and per-source node and edge provenance.
- Added incremental document synchronization for additions, changes, and deletions.
- Added source-filtered knowledge graph projections.
- Added semantic entity resolution using embedding candidates and LLM-assisted
  merge decisions when combining document graphs.
- Added cached graph and chunk embeddings with batched embedding requests.
- Added JSONL audit logging for generated edges that reference missing nodes.
- Added a canonical node taxonomy with persisted suggestions for unclassified
  node types.
- Added hybrid BM25 and semantic chunk retrieval.
- Added a LangGraph-based question-answering workflow with source reporting and
  retrieval traces.
- Added centralized runtime, path, client, and store construction.

### Changed

- Replaced the single Azure OpenAI model configuration with separate flagship,
  fast, and embedding deployment settings.
- Added configuration for embedding batch size and the entity-resolution
  candidate similarity threshold.
- Improved graph extraction to exclude document-structure and metadata nodes,
  resolve repeated entities, and deduplicate relationships.
- Kept stored graphs, chunks, embeddings, taxonomy suggestions, and generated
  visualizations synchronized with document additions, updates, and deletions.
- Updated the main workflow to synchronize documents, export visualizations,
  and answer questions through the shared runtime.

### Fixed

- Scoped generated node references to their source chunks to prevent collisions.
- Validated unique extracted node references and retried invalid structured LLM
  responses once.

### Removed

- Removed the single-text `EmbeddingClient.embed_text` API in favor of batched
  `embed_texts` calls.
- Removed unused model metadata from taxonomy suggestion records. Existing
  `data/taxonomy_suggestions.json` files must be deleted and regenerated.

## [0.6.0]

### Added

- Added run logging with log rotation.
- Added cached combined knowledge base graph storage on disk.

## [0.5.0]

### Added

- Added PyVis-based HTML knowledge graph visualization export.
- Added knowledge base visualization helpers.

### Changed

- Extracted corpus loading and visualization workflow out of `main.py`.

## [0.4.0]

### Added

- Added multi-document markdown loading.
- Added per-document graph and chunk storage.
- Added source-key handling for stable document identifiers.
- Added skipping for already-processed documents.

## [0.3.0]

### Added

- Added BM25 retrieval over graph nodes, relationships, and chunks.
- Added formatted retrieval context with source tracking.

## [0.2.0]

### Added

- Added knowledge graph node, edge, metadata, and graph models.
- Added JSON graph storage.
- Added LLM-based graph generation.
- Added markdown chunking and chunk JSON storage.
- Added chunk provenance on generated nodes and edges.

### Changed

- Tightened graph extraction schema used with the LLM.

## [0.1.0]

### Added

- Added initial Python project structure.
- Added Azure OpenAI client abstraction.
- Added Azure OpenAI configuration loading.
- Added initial package metadata and tests.
