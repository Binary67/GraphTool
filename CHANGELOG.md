# Changelog

## [Unreleased]

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
