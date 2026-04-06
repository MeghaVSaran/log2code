# Pathless Boosting Round 2 Plan

## Objective

Improve no-explicit-path Phase 1 retrieval by converting build/object traces into usable source-path hints and using those hints during candidate generation and ranking.

## Planned Work

1. Expand source path extraction to include object-file traces and CMake object paths.
2. Use build-directory context to recover repository-relative paths from partial object paths.
3. Feed normalized hint/source paths into retrieval as additional sparse candidates.
4. Add component-prefix context boost in retriever scoring.
5. Ensure ablation/eval pass repo root to log parser so normalized paths are available.
6. Add regression tests for all new no-path extraction patterns.
