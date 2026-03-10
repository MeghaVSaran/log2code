# Vector Index — Implementation Plan

## Goal
Implement `src/indexing/vector_index.py` wrapping ChromaDB PersistentClient for dense retrieval.

## Proposed Changes

### [MODIFY] vector_index.py (was stub)
- `__init__`: creates `PersistentClient(path=str(persist_dir))`, sets `_collection = None`
- `build()`: delete existing collection if present → create fresh with `hnsw:space=cosine` → upsert chunks with metadata in batches of 5000
- `query()`: calls `_ensure_collection()` → `collection.query()` → converts `1 - distance` to score
- `exists()`: tries `get_collection()` + `count() > 0`
- `_ensure_collection()`: loads collection or raises `IndexNotFoundError`

### [NEW] test_vector_index.py
11 tests using `FakeChunk` dataclass and random numpy embeddings:
- Build creates/overwrites collection
- Query returns correct fields, top-1 self-match, proper score range, respects top_k, accepts numpy
- Exists before/after build
- IndexNotFoundError on query before build
- Persistence across VectorIndex instances
