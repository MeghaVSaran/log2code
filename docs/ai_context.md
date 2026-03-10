# DebugAid — AI Coding Context

## Read This First

You are helping build DebugAid, an ML-powered CLI tool that maps C++ build/runtime
error logs to the most relevant source files and functions in a codebase.

This file is the single source of truth for all AI-assisted coding sessions.
Always follow the architecture and conventions defined here.

---

## Project Summary

**What it does:**
Given a C++ error log and a C++ repository, return the top 5 source files and
functions most likely responsible for the error.

**How it works:**
1. C++ codebase is parsed into function-level chunks using Tree-sitter
2. Each chunk is embedded using GraphCodeBERT (768-dim vectors)
3. Chunks are stored in ChromaDB (dense) and BM25 index (sparse)
4. Error log is parsed with regex to extract error type and identifiers
5. Log is embedded using all-mpnet-base-v2
6. Hybrid retrieval: 0.6 * dense_score + 0.4 * bm25_score
7. Top 5 results returned with file path, function name, line number, score

---

## Repository Structure

```
debugaid/
  docs/                         ← architecture and spec docs (read-only)
  src/
    ingestion/
      code_parser.py            ← Tree-sitter C++ → function chunks
      log_parser.py             ← regex log → structured error info
    embeddings/
      code_embedder.py          ← GraphCodeBERT embeddings for code
      log_embedder.py           ← all-mpnet-base-v2 embeddings for logs
    indexing/
      vector_index.py           ← ChromaDB wrapper
      bm25_index.py             ← rank_bm25 wrapper
    retrieval/
      hybrid_retriever.py       ← fuses dense + sparse results
    evaluation/
      metrics.py                ← Recall@K, MRR
      benchmark.py              ← runs eval on ground truth dataset
    cli/
      main.py                   ← click CLI: index, query, eval, info
  scripts/
    mine_github_issues.py       ← dataset: mine GitHub for labeled pairs
    generate_synthetic_errors.py ← dataset: inject C++ errors, compile, capture
  tests/
    test_log_parser.py
    test_code_parser.py
    test_hybrid_retriever.py
    test_metrics.py
  data/
    ground_truth/
      dataset.json              ← labeled (log, relevant_files) pairs
  requirements.txt
  setup.py
  README.md
```

---

## Tech Stack (Do Not Deviate)

| Component | Library/Model | Notes |
|---|---|---|
| Code parsing | `tree-sitter`, `tree-sitter-cpp` | Function-level chunks only |
| Code embeddings | `microsoft/graphcodebert-base` | HuggingFace transformers |
| Log embeddings | `sentence-transformers/all-mpnet-base-v2` | 768-dim output |
| Vector store | `chromadb` | Persistent mode |
| Sparse retrieval | `rank_bm25` | BM25Okapi |
| CLI | `click` | |
| Testing | `pytest` | |
| Data | `pandas`, `numpy` | |

---

## Core Data Structures

### Chunk (output of code_parser)
```python
@dataclass
class Chunk:
    chunk_id: str          # "{file_path}::{function_name}"
    file_path: str         # relative to repo root
    function_name: str     # e.g. "Parser::resolveSymbol"
    start_line: int
    end_line: int
    code_text: str         # full function source text
    language: str = "cpp"
```

### ParsedLog (output of log_parser)
```python
@dataclass
class ParsedLog:
    raw_log: str
    error_type: str        # "linker_error" | "compiler_error" | "include_error" | "template_error" | "segfault"
    error_message: str     # single most informative line
    identifiers: List[str] # extracted symbol names
    file_hints: List[str]  # any filenames mentioned in log
    stack_frames: List[str]
```

### RetrievalResult (output of hybrid_retriever)
```python
@dataclass
class RetrievalResult:
    rank: int
    chunk_id: str
    file_path: str
    function_name: str
    start_line: int
    score: float
    dense_score: float
    bm25_score: float
```

---

## Coding Conventions

- Python 3.11+
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- Dataclasses for all data structures
- No global state — pass dependencies explicitly
- All file I/O uses pathlib.Path
- Logging via Python `logging` module (not print statements in library code)
- CLI is allowed to print to stdout
- Tests use pytest, no unittest
- Each module is independently importable (no circular imports)

---

## Error Handling Rules

- `code_parser.py`: if a file fails to parse, log a warning and return []
- `log_parser.py`: always return a ParsedLog even if parsing is partial
- `vector_index.py`: raise IndexNotFoundError if query called before build
- `hybrid_retriever.py`: if one index fails, fall back to the other (log warning)
- CLI: all exceptions caught at top level, print friendly message, exit 1

---

## What NOT to Build

Do not add any of these to the codebase — they are Phase 2:
- Cross-encoder re-ranking
- LLM explanation generation
- Streamlit or any web UI
- UMAP / HDBSCAN clustering
- libclang integration
- Incremental index updates
- Support for languages other than C++

If you are asked to implement any of these, refuse and note they are Phase 2.

---

## Evaluation Targets

The system is considered working when:
- Recall@5 >= 0.65 on dev set
- MRR >= 0.45 on dev set
- Works on: linker errors, compiler errors, include errors, segfaults

---

## How to Ask for Code

Good prompts to use:
```
Implement src/ingestion/code_parser.py following the conventions in docs/ai_context.md.
The function parse_repository(repo_path: Path) -> List[Chunk] should walk all C++ files
and extract function-level chunks using Tree-sitter.
Include docstrings and pytest-compatible unit tests.
```

Always specify:
- Which file you want implemented
- Which function(s) to implement
- Any specific edge cases to handle
- Whether tests are needed

Never ask for "the whole system" in one prompt.
