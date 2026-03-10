# MVP Specification

## MVP Goal

A working CLI tool that takes a C++ build/runtime log and a C++ repository,
and returns the top 5 most likely source files and functions responsible for
the error — with measurable accuracy on a ground truth dataset.

## MVP Is Complete When

- [ ] `debugaid index --repo ./some-cpp-project` runs without errors
- [ ] `debugaid query --log build.log` returns ranked results in < 3 seconds
- [ ] Recall@5 >= 0.65 on ground truth dataset
- [ ] MRR >= 0.45 on ground truth dataset
- [ ] Works correctly on at least 4 error categories (linker, compiler, include, segfault)
- [ ] Tested on LLVM or OpenCV (real large codebase)

---

## Features IN the MVP

### F1 — Code Ingestion
- Parse C++ repository using Tree-sitter
- Extract function-level chunks (function name, file, start line, end line, code text)
- Handle .cpp, .cc, .cxx, .h, .hpp files
- Skip files that fail to parse (log warning, continue)
- Output chunk list as JSON for inspection

### F2 — Code Indexing
- Embed all chunks using GraphCodeBERT
- Store vectors + metadata in ChromaDB (persistent on disk)
- Build BM25 index from chunk texts
- Save both indices to a `.debugaid/` folder inside the repo

### F3 — Log Parsing
- Regex-based parser covering 5 error categories:
  1. Linker errors (undefined reference, multiple definition)
  2. Compiler errors (undeclared identifier, no matching function)
  3. Include errors (no such file or directory)
  4. Template errors (implicit instantiation, undefined template)
  5. Segfault stack traces (#N frame lines)
- Extract: error_type, identifiers, file_hints, error_message
- Handle multi-line logs (extract the most informative lines)

### F4 — Hybrid Retrieval
- Embed log using all-mpnet-base-v2
- Dense search via ChromaDB (top 20 candidates)
- Sparse search via BM25 (top 20 candidates)
- Normalize and fuse scores: `0.6 * dense + 0.4 * bm25`
- Return top 5 results

### F5 — CLI Interface
```
debugaid index  --repo PATH [--force-reindex]
debugaid query  --log PATH --repo PATH [--top-k N] [--verbose]
debugaid eval   --dataset PATH
debugaid info   --repo PATH
```

### F6 — Evaluation Harness
- Load ground truth JSON dataset
- Run pipeline on each entry
- Compute and display:
  - Recall@1, Recall@3, Recall@5
  - MRR (Mean Reciprocal Rank)
  - Per-error-type breakdown

### F7 — Ground Truth Dataset
- 500+ labeled pairs: (log_snippet, relevant_files[])
- Sources: GitHub issue mining + synthetic compiler errors
- Covers all 5 error categories
- Stored in `data/ground_truth/dataset.json`

---

## Features NOT in MVP (Phase 2)

| Feature | Why Deferred |
|---|---|
| Cross-encoder re-ranking | Extra model, extra complexity |
| LLM-generated explanations | Optional UX feature, not core |
| Streamlit web UI | CLI is sufficient for demo |
| UMAP + HDBSCAN clustering | Not core to retrieval quality |
| libclang semantic analysis | Tree-sitter sufficient for MVP |
| Incremental index updates | Full reindex acceptable for MVP |
| Support for other languages | C++ only for now |

---

## MVP Module Checklist

### `src/ingestion/code_parser.py`
- [ ] `parse_repository(repo_path) -> List[Chunk]`
- [ ] `parse_file(file_path) -> List[Chunk]`
- [ ] Handles all C++ file extensions
- [ ] Skips unparseable files gracefully
- [ ] Returns empty list for non-C++ files

### `src/ingestion/log_parser.py`
- [ ] `parse_log(log_text) -> ParsedLog`
- [ ] `extract_error_type(log_text) -> str`
- [ ] `extract_identifiers(log_text) -> List[str]`
- [ ] `extract_file_hints(log_text) -> List[str]`
- [ ] Handles multi-line logs
- [ ] Returns structured dict even if parsing is partial

### `src/embeddings/code_embedder.py`
- [ ] `embed_chunks(chunks, batch_size=16) -> List[Vector]`
- [ ] Uses GraphCodeBERT
- [ ] Formats input as: `<function> NAME <context> FILE\nCODE`
- [ ] Handles chunks longer than model max length (truncate)

### `src/embeddings/log_embedder.py`
- [ ] `embed_log(parsed_log) -> Vector`
- [ ] Uses all-mpnet-base-v2
- [ ] Input = `error_message + " " + " ".join(identifiers)`

### `src/indexing/vector_index.py`
- [ ] `build(chunks, embeddings, persist_dir)`
- [ ] `query(log_embedding, top_k=20) -> List[Result]`
- [ ] `load(persist_dir)`
- [ ] `exists(persist_dir) -> bool`

### `src/indexing/bm25_index.py`
- [ ] `build(chunks)`
- [ ] `query(text, top_k=20) -> List[Result]`
- [ ] `save(path)` / `load(path)`

### `src/retrieval/hybrid_retriever.py`
- [ ] `retrieve(log_embedding, log_text, top_k=5) -> List[RankedResult]`
- [ ] `_normalize_scores(scores) -> List[float]`
- [ ] `_fuse_scores(dense, sparse, alpha=0.6) -> Dict[chunk_id, float]`

### `src/evaluation/metrics.py`
- [ ] `recall_at_k(predictions, ground_truth, k) -> float`
- [ ] `mrr(predictions, ground_truth) -> float`
- [ ] `evaluate_dataset(dataset, retriever) -> EvalReport`

### `src/cli/main.py`
- [ ] `index` command
- [ ] `query` command
- [ ] `eval` command
- [ ] `info` command
- [ ] Human-readable output formatting

---

## Acceptance Test Cases

### Test 1 — Linker Error
**Input log:**
```
/usr/bin/ld: CMakeFiles/llvm.dir/lib/CodeGen/MachineFunction.cpp.o: undefined reference to `llvm::TargetRegisterInfo::getRegAsmName(unsigned int) const'
collect2: error: ld returned 1 exit status
```
**Expected:** `lib/Target/TargetRegisterInfo.cpp` in top 3.

### Test 2 — Missing Include
**Input log:**
```
fatal error: llvm/Support/CommandLine.h: No such file or directory
#include "llvm/Support/CommandLine.h"
```
**Expected:** `include/llvm/Support/CommandLine.h` or a file that defines its contents in top 5.

### Test 3 — Segfault Stack Trace
**Input log:**
```
Segmentation fault (core dumped)
#0  0x00007f llvm::SelectionDAG::getNode()
#1  0x00007f llvm::X86TargetLowering::LowerOperation()
```
**Expected:** Files containing `SelectionDAG::getNode` or `LowerOperation` in top 3.

### Test 4 — Undeclared Identifier
**Input log:**
```
error: use of undeclared identifier 'createTargetMachine'
       auto TM = createTargetMachine(Triple, CPU, Features);
```
**Expected:** File where `createTargetMachine` is declared/defined in top 5.

---

## Performance Targets

| Metric | Target |
|---|---|
| Indexing speed | < 10 min for 500k LOC |
| Query latency | < 3 seconds |
| Recall@5 | >= 0.65 |
| MRR | >= 0.45 |
| Min dataset size | 500 labeled pairs |
