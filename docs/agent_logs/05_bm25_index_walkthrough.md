# BM25 Index — Walkthrough

## Changes Made

### bm25_index.py (was stub → fully implemented)
- Tokenizer: `re.compile(r"[\s()\{\};,<>*&:.]+|->|::")` splits C++ code cleanly
- `build()`: tokenizes code_text per chunk, fits `BM25Okapi`
- `query()`: stops at zero-score results, returns empty for no-match
- `save()`/`load()`: pickle dict with `{index, chunks}`

### test_bm25_index.py (new)
4 test classes, 13 test methods.

## Verification — 13/13 passed ✅ (0.43s)

```
TestBuildAndQuery  — 4 passed
TestNoMatch        — 3 passed
TestSaveLoad       — 2 passed
TestTokenize       — 4 passed
```
