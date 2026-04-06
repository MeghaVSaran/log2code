# Phase 1 Adaptive Fusion and Component Expansion Plan

## Objective

Improve difficult no-explicit-path retrieval cases by:

1. Expanding candidates from component path context (not only direct file hints),
2. Adapting dense vs sparse fusion weights to error type and query structure.

## Planned Changes

1. Add BM25 `query_by_path_prefix()` to retrieve candidates from likely component subtrees.
2. Integrate prefix-based candidate expansion into hybrid retrieval hint path.
3. Add adaptive fusion weight selection using parsed log signals (`error_type`, identifiers, stack frames, hints).
4. Add tests for:
   - prefix-query behavior,
   - adaptive weighting preference for lexical linker signals.
