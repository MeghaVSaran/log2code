# Phase 1 Adaptive Fusion and Component Expansion Walkthrough

## Summary

This pass improves retrieval behavior for hard linker/pathless logs by broadening candidate generation and making fusion weights query-aware.

## Implemented Changes

- `src/indexing/bm25_index.py`
  - Added `query_by_path_prefix(prefixes, top_k)` for component-level candidate retrieval.
  - This helps retrieve files under directories like `absl/base/*` even when the log only contains partial build path signals.

- `src/retrieval/hybrid_retriever.py`
  - Added prefix candidate expansion in `_get_hint_candidates()` by calling `query_by_path_prefix()`.
  - Added `_select_fusion_weights(parsed_log)`:
    - lexical-heavy error types (`compiler_error`, `linker_error`, `include_error`, etc.) now prefer sparse scores more strongly,
    - runtime/stack-trace classes use a balanced-but-sparse-leaning mix,
    - path hints increase sparse emphasis further.
  - `_fuse()` now accepts per-query dense/sparse weights.

- Tests
  - `tests/test_bm25_index.py`: added path-prefix query test.
  - `tests/test_hybrid_retriever.py`: added adaptive-fusion test ensuring linker lexical matches can outrank dense noise.

## Verification

Executed:

```powershell
pytest -q
```

Result:

- `180 passed in 6.47s`

## Expected Impact

1. Better top-rank behavior on linker/compiler queries where exact identifiers dominate.
2. Better recall for files in the same component subtree as object/build hints.
3. Stronger practical Phase 1 query performance without introducing heavy reranker models.
