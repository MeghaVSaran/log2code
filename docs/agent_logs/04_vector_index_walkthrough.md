# Vector Index — Walkthrough

## Changes Made

### vector_index.py (was stub → fully implemented)
- ChromaDB `PersistentClient` with cosine distance space
- `build()`: delete + create collection, batch upsert in groups of 5000
- `query()`: nearest-neighbour search, `score = 1 - distance`
- `exists()`: `get_collection().count() > 0`
- Compatible with chromadb 1.5.5 (catches `Exception` for missing collections)

### test_vector_index.py (new)
5 test classes, 11 test methods using random numpy embeddings.

## Verification

### pytest — 11/11 passed ✅ (6.25s)

```
TestBuild                 — 2 passed
TestQuery                 — 5 passed
TestExists                — 2 passed
TestIndexNotFoundError    — 1 passed
TestPersistence           — 1 passed
```

### Bug fix during testing
chromadb 1.5.5 raises `NotFoundError` (not `ValueError`) when deleting a non-existent collection. Fixed by catching `Exception`.
