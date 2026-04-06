# Phase 1 Pathless Hardening Plan

## Goal

Improve DebugAid Phase 1 retrieval quality for logs without explicit source paths, especially linker and stack-trace-driven debugging.

## Problems Found

1. `run_ablation.py` did not pass `parsed_log` into retriever calls, so symbol-aware retrieval logic was not measured.
2. BM25 query logic dropped non-positive scores, which can suppress valid ranking when BM25 returns negative scores.
3. Symbol boosting in retriever only reranked already-retrieved candidates and could not recover missed candidates.
4. Segfault stack-frame extraction in `log_parser.py` was too narrow for many real crash formats.
5. Source-path extraction did not robustly handle both rooted and relative path formats.

## Planned Fixes

1. Expand stack-frame parsing and source-path normalization in `log_parser.py`.
2. Add symbol lookup index and `query_by_symbols()` in `bm25_index.py`.
3. Merge symbol-derived candidates into sparse candidates before hybrid fusion in `hybrid_retriever.py`.
4. Improve score normalization edge cases in retriever.
5. Fix ablation runner to pass parsed logs and add `--repo-filter` plus dataset-repo mismatch warnings.
6. Add tests for new parser, BM25, and retriever behavior.
