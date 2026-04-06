# Symbol-Aware Retrieval Walkthrough

## What Changed

This implementation strengthens DebugAid's existing `query` path for logs that mention symbols but not explicit source paths.

- `src/ingestion/code_parser.py`
  - Added chunk metadata fields: `symbol_name`, `class_name`, `namespace`, `signature`
  - Derived symbol metadata from qualified function names
  - Captured a compact signature string for each extracted function
- `src/indexing/bm25_index.py`
  - Indexed augmented search text instead of raw code only
  - Returned symbol metadata alongside retrieval hits
- `src/indexing/vector_index.py`
  - Persisted the new symbol metadata into ChromaDB
  - Returned it on query so retrieval can reason over it
- `src/embeddings/code_embedder.py`
  - Formatted dense embedding input with explicit function, class, namespace, path, and signature context
- `src/retrieval/hybrid_retriever.py`
  - Added a symbol-aware boost stage driven by parsed identifiers and stack frames
  - Preserved existing hybrid fusion and source-path injection behavior
  - Extended `RetrievalResult` with `symbol_score`
- `src/cli/main.py`
  - Threaded parsed-log context into retrieval
  - Exposed `symbol_score` in verbose and JSON output
- `src/evaluation/metrics.py`
  - Threaded parsed-log context into retrieval during evaluation
- `tests/test_code_parser.py`
  - Added metadata assertions for qualified and inline methods
- `tests/test_hybrid_retriever.py`
  - Added tests covering exact-symbol and stack-frame boosts

## Why This Helps

The current-system summary showed the core failure mode clearly: when logs do not contain direct file paths, retrieval collapses. These changes make the existing pipeline more symbol-aware without replacing the measurable retrieval core.

That gives DebugAid a stronger answer for cases like:

- `undefined reference to Parser::resolveSymbol`
- segfault traces naming `Parser::resolveSymbol`
- logs where the most useful clue is a function or method name rather than a path

## Verification

Ran:

```powershell
pytest tests/test_code_parser.py tests/test_hybrid_retriever.py tests/test_bm25_index.py tests/test_metrics.py tests/test_vector_index.py -q
```

Result:

- `102 passed in 9.51s`

## Notes

- This is still an evidence-weighting upgrade, not a full AST reasoning layer.
- The next highest-impact follow-up is to evaluate this against the no-source-path bucket specifically and tune the symbol boost weights using that slice rather than overall averages.
