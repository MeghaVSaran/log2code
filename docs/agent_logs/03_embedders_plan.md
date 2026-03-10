# Embedders — Implementation Plan

## Goal
Implement both embedding modules: `code_embedder.py` (GraphCodeBERT) and `log_embedder.py` (all-mpnet-base-v2).

## Proposed Changes

### [MODIFY] code_embedder.py (was stub)
- `__init__`: stores model_name and device, sets `_model = None` (lazy)
- `_ensure_model()`: loads AutoTokenizer + AutoModel on first call
- `embed_chunks(chunks, batch_size=16)`: formats chunks, processes in batches
- `_embed_batch()`: tokenize → forward pass → mean pool over last_hidden_state with attention mask
- `embed_text()`: single-string convenience method
- `_format_chunk()`: `<function> {name} <context> {path}\n{code}`
- `__main__` block with sample Chunk

### [MODIFY] log_embedder.py (was stub)
- `__init__`: stores model_name, sets `_model = None` (lazy)
- `_ensure_model()`: loads SentenceTransformer on first call
- `embed_log(parsed_log)`: calls `parsed_log.query_text()` → `embed_text()`
- `embed_text()`: `model.encode(text, convert_to_numpy=True).tolist()`
- `__main__` block with sample log

## Verification
- Import-only test: confirm `__init__` does NOT trigger model download
- No pytest tests (require model downloads)
