# Phase 1 Pathless Hardening Walkthrough

## Summary

This pass fixed concrete retrieval-quality bugs and strengthened pathless candidate recall for Phase 1.

## Key Changes

- `src/ingestion/log_parser.py`
  - Broadened stack-frame regex to capture common real-world forms (`#0 0x... Symbol(...)` and `#0 ... in Symbol(...)`).
  - Included stack-frame pattern in segfault error-message selection.
  - Added optional `repo_root` to `parse_log()` and populated `ParsedLog.source_paths`.
  - Improved source path extraction/normalization for rooted and relative paths.

- `src/indexing/bm25_index.py`
  - Fixed BM25 bug that previously dropped non-positive scores, causing empty results in some valid queries.
  - Added symbol lookup index at build time.
  - Added `query_by_symbols()` for exact/near-exact symbol candidate retrieval.
  - Persisted and restored symbol lookup in save/load.

- `src/retrieval/hybrid_retriever.py`
  - Added symbol-candidate retrieval before fusion (instead of only post-fusion reranking).
  - Merged symbol candidates into sparse candidates by chunk id with max score.
  - Improved normalization for equal-score edge cases (`[1.0, ...]` instead of all zeros).

- `src/evaluation/run_ablation.py`
  - Passed `parsed_log` into `retriever.retrieve(...)`, enabling symbol-aware logic during ablation.
  - Added `symbol_score` to per-sample score traces.
  - Added `--repo-filter` to evaluate only in-repo samples.
  - Added warning when many dataset samples are out of indexed-repo scope.

- Tests
  - `tests/test_log_parser.py`: stack frames without `in`, relative source-path extraction.
  - `tests/test_bm25_index.py`: symbol query behavior and negative-score regression.
  - `tests/test_hybrid_retriever.py`: symbol candidates entering recall set and normalization edge-case checks.

## Verification

Executed:

```powershell
pytest -q
```

Result:

- `175 passed in 5.48s`

## Expected Impact

1. Pathless queries now gain candidate recall from symbol-index matches, not only score boosts.
2. Segfault/stack traces should extract more usable symbols in real logs.
3. Ablation outputs now reflect the actual current retriever behavior and can be fairly scoped to an indexed repo.
