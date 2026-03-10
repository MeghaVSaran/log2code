# Hybrid Retriever — Implementation Plan

## Goal
Implement `src/retrieval/hybrid_retriever.py` fusing dense (ChromaDB) and sparse (BM25) retrieval.

## Changes

### [MODIFY] hybrid_retriever.py (was stub)
- `retrieve()`: queries both indices (top 20 each), normalizes, fuses, sorts, returns top_k
- `_normalize_scores()`: min-max to [0,1]
- `_fuse()`: merges unique chunk_ids, applies 0.6*dense + 0.4*bm25 weights
- Fallback: catches IndexNotFoundError for dense-only failure, empty BM25 result set

### [NEW] test_hybrid_retriever.py
13 tests with FakeVectorIndex/FakeBM25Index: fusion math, sorting, fallback, result types
