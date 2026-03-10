# Embedders — Walkthrough

## Changes Made

### code_embedder.py (was stub → fully implemented)
- Lazy loading: `_ensure_model()` defers transformers/torch imports until first embed call
- Mean pooling: attention-mask-weighted mean over `last_hidden_state`
- Batching: processes chunks in groups of `batch_size` (default 16)
- Input format: `<function> {name} <context> {path}\n{code}`

### log_embedder.py (was stub → fully implemented)
- Lazy loading: `_ensure_model()` defers SentenceTransformer import
- Uses `model.encode()` which handles tokenization and pooling internally
- Input: `parsed_log.query_text()` → `error_message + identifiers`

## Verification

### Lazy loading confirmed ✅
```
CodeEmbedder init OK, model NOT loaded: True
LogEmbedder init OK, model NOT loaded: True
```

### Existing tests still pass ✅
All 51 tests across log_parser and code_parser pass.
