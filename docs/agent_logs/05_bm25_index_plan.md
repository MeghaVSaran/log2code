# BM25 Index — Implementation Plan

## Goal
Implement `src/indexing/bm25_index.py` using rank_bm25 for sparse retrieval.

## Proposed Changes

### [MODIFY] bm25_index.py (was stub)
- `_tokenize()`: regex split on C++ operators/punctuation, filter <2 chars, filter numeric, lowercase
- `build()`: tokenize each chunk's code_text, fit BM25Okapi
- `query()`: tokenize query → get_scores → argsort descending → return top_k (stop at zero)
- `save()`/`load()`: pickle dict of `{index, chunks}`

### [NEW] test_bm25_index.py
13 tests: build+query, identifier matching, no-match, save/load round-trip, tokenizer behaviour
