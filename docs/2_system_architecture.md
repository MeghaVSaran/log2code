# System Architecture

## High-Level Pipeline

```
C++ Repository
      │
      ▼
┌─────────────────┐
│  Code Ingestion  │  Tree-sitter parses all .cpp/.h files
│                 │  Extracts function-level chunks
│                 │  Output: {file, function, start_line, end_line, code_text}
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Code Embedding │  GraphCodeBERT converts each chunk to 768-dim vector
│                 │  Stored in ChromaDB with metadata
│                 │  Also indexed in BM25 on raw text
└────────┬────────┘
         │
         ▼
┌─────────────────┐         ┌──────────────────┐
│  Vector Index   │         │   BM25 Index     │
│  (ChromaDB)     │         │   (rank_bm25)    │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         └──────────┬────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Hybrid Retriever │  score = 0.6 * dense + 0.4 * bm25
         └──────────┬───────┘
                    │
         ┌──────────┴───────┐
         │                  │
         ▼                  ▼
┌──────────────┐   ┌──────────────────┐
│  Log Parser  │   │  Log Embedder    │
│  (regex)     │   │  all-mpnet-base  │
└──────┬───────┘   └────────┬─────────┘
       │                    │
       └──────────┬─────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Ranked Output  │  Top-5 files + functions + scores
         └────────────────┘
```

## Component Breakdown

### 1. Code Ingestion (`src/ingestion/code_parser.py`)

**Responsibility:** Parse a C++ repository into function-level chunks.

**Input:** Path to a C++ repository.

**Output:** List of chunk objects:
```python
{
    "chunk_id": "src/parser/resolve.cpp::Parser::resolveSymbol",
    "file_path": "src/parser/resolve.cpp",
    "function_name": "Parser::resolveSymbol",
    "start_line": 142,
    "end_line": 168,
    "code_text": "void Parser::resolveSymbol(Symbol &s) { ... }",
    "language": "cpp"
}
```

**How:** Tree-sitter walks the AST and extracts nodes of type:
- `function_definition`
- `function_declarator`
- `class_specifier` (for class-level context)

**Edge cases handled:**
- Templated functions
- Anonymous namespaces
- Inline functions in headers
- Files with parse errors (skip and log, do not crash)

---

### 2. Log Parser (`src/ingestion/log_parser.py`)

**Responsibility:** Extract structured error information from raw log text.

**Input:** Raw log string (GCC, Clang, linker, runtime).

**Output:**
```python
{
    "raw_log": "...",
    "error_type": "linker_error",
    "identifiers": ["Parser::resolveSymbol"],
    "file_hints": ["resolve.cpp"],
    "error_message": "undefined reference to Parser::resolveSymbol",
    "stack_frames": []
}
```

**Error categories handled:**

| Category         | Example                                              | Pattern                          |
|------------------|------------------------------------------------------|----------------------------------|
| Linker error     | `undefined reference to Parser::resolveSymbol`       | `undefined reference to`         |
| Linker error     | `multiple definition of SymbolTable::insert`         | `multiple definition of`         |
| Compiler error   | `no matching function for call to Parser::parse`     | `no matching function`           |
| Compiler error   | `use of undeclared identifier resolveSymbol`         | `undeclared identifier`          |
| Include error    | `fatal error: parser/resolve.h: No such file`        | `fatal error:.*No such file`     |
| Template error   | `error: implicit instantiation of undefined template`| `implicit instantiation`         |
| Segfault trace   | `#0 Parser::resolveSymbol (this=...)`                | `#\d+\s+\w+::\w+`               |
| Assert failure   | `Assertion failed: sym != nullptr`                   | `Assertion.*failed`              |

---

### 3. Code Embedder (`src/embeddings/code_embedder.py`)

**Model:** `microsoft/graphcodebert-base`

**Input:** List of chunk objects.

**Output:** 768-dimensional float vectors, one per chunk.

**Batching:** Process in batches of 16 to avoid OOM on CPU.

**Text format passed to model:**
```
<function> Parser::resolveSymbol <context> src/parser/resolve.cpp
void Parser::resolveSymbol(Symbol &s) {
    ...
}
```

---

### 4. Log Embedder (`src/embeddings/log_embedder.py`)

**Model:** `sentence-transformers/all-mpnet-base-v2`

**Input:** Parsed log object (use `error_message` + `identifiers` joined).

**Output:** 768-dimensional float vector.

**Note:** all-mpnet-base-v2 outputs 768 dims, matching GraphCodeBERT. This
avoids any dimension mismatch in the retrieval layer.

---

### 5. Vector Index (`src/indexing/vector_index.py`)

**Tool:** ChromaDB (persistent mode).

**Collection name:** `debugaid_code_chunks`

**Stored per chunk:**
- embedding vector
- metadata: file_path, function_name, start_line, end_line, chunk_id

**Key operations:**
- `build(chunks, embeddings)` — create or overwrite index
- `query(log_embedding, top_k)` — return top_k results with scores
- `update(new_chunks, new_embeddings)` — add without rebuilding

---

### 6. BM25 Index (`src/indexing/bm25_index.py`)

**Tool:** `rank_bm25` (BM25Okapi).

**Input:** List of tokenized code texts (same chunks as vector index).

**Tokenization:** Split on whitespace + strip C++ operators. Keep identifiers intact.

**Key operations:**
- `build(chunks)` — fit BM25 on corpus
- `query(log_text, top_k)` — return top_k chunk_ids with scores
- `save(path)` / `load(path)` — pickle-based persistence

---

### 7. Hybrid Retriever (`src/retrieval/hybrid_retriever.py`)

**Inputs:**
- log embedding (from log embedder)
- log text (from log parser — raw error message + identifiers)
- top_k (default 5)

**Process:**
1. Query ChromaDB with log embedding → get top 20 dense results with scores
2. Query BM25 with log text → get top 20 sparse results with scores
3. Normalize both score lists to [0, 1]
4. For each unique chunk_id: `final_score = 0.6 * dense + 0.4 * bm25`
5. Sort by final_score, return top_k

**Output:**
```python
[
    {
        "rank": 1,
        "chunk_id": "src/parser/resolve.cpp::Parser::resolveSymbol",
        "file_path": "src/parser/resolve.cpp",
        "function_name": "Parser::resolveSymbol",
        "start_line": 142,
        "score": 0.91
    },
    ...
]
```

---

### 8. Evaluation (`src/evaluation/`)

**metrics.py** — computes Recall@K and MRR given predictions and ground truth.

**benchmark.py** — loads ground truth dataset, runs pipeline on each entry,
computes aggregate metrics, prints report.

**Ground truth format:**
```json
{
    "log": "undefined reference to Parser::resolveSymbol",
    "relevant_files": ["src/parser/resolve.cpp", "include/parser/resolve.h"]
}
```

---

### 9. CLI (`src/cli/main.py`)

Built with `click`.

```
debugaid index   --repo ./llvm-project
debugaid query   --log build.log --repo ./llvm-project [--top-k 5]
debugaid eval    --dataset data/ground_truth/dataset.json
debugaid info    --repo ./llvm-project
```

---

## Data Flow Summary

```
INDEXING TIME (run once per repo):
repo → code_parser → chunks → code_embedder → vectors → chroma + bm25

QUERY TIME (run per log):
log_file → log_parser → structured_log → log_embedder → log_vector
log_vector + log_text → hybrid_retriever → ranked results → CLI output
```

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Chunk granularity | Function-level | File-level too coarse, line-level too noisy |
| Embedding model for code | GraphCodeBERT | Trained on code, understands structure |
| Embedding model for logs | all-mpnet-base-v2 | Best general semantic model at 768d |
| Vector DB | ChromaDB | Persistent, metadata filtering, simple API |
| Sparse retrieval | BM25 | Exact identifier matching; cheap to run |
| Fusion weights | 0.6 dense / 0.4 BM25 | Dense for semantics, BM25 for identifiers |
| Interface | CLI | Demonstrable, composable, CI-friendly |
