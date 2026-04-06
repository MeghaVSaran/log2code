# Claude Code Prompts — Copy-Paste Ready

Use these prompts IN ORDER. Each one builds on the previous.
Always start a new Claude Code session by saying:
"Read docs/ai_context.md before writing any code."

---

## PROMPT 1 — Log Parser (Start Here)

```
Read docs/ai_context.md and docs/2_system_architecture.md section 2 (Log Parser).

Implement src/ingestion/log_parser.py completely.

Requirements:
- Implement parse_log(log_text: str) -> ParsedLog
- Implement extract_error_type(log_text: str) -> str
- Implement extract_identifiers(log_text: str) -> List[str]
- Implement extract_file_hints(log_text: str) -> List[str]

Handle exactly these 5 error categories with regex patterns:
1. linker_error: "undefined reference to X", "multiple definition of X"
2. compiler_error: "use of undeclared identifier X", "no matching function for call to X"
3. include_error: "fatal error: X.h: No such file or directory"
4. template_error: "implicit instantiation of undefined template", "undefined template"
5. segfault: "Segmentation fault", stack frames like "#0  ClassName::methodName"

For identifiers, extract the symbol name after "undefined reference to",
"undeclared identifier", "multiple definition of", and from stack frame lines.
Strip C++ mangling suffixes where possible.

ParsedLog.query_text() should return: error_message + " " + " ".join(identifiers)

Write complete implementation (no stubs).
Then write tests/test_log_parser.py with pytest tests covering:
- One test per error category
- Multi-line log handling
- Log with no recognizable error (returns error_type="unknown")
- Log with multiple error lines (returns most specific one)
```

---

## PROMPT 2 — Code Parser

```
Read docs/ai_context.md and docs/2_system_architecture.md section 1 (Code Ingestion).

Implement src/ingestion/code_parser.py completely.

Requirements:
- Implement parse_repository(repo_path: Path) -> List[Chunk]
- Implement parse_file(file_path: Path, repo_root: Path) -> List[Chunk]

Use tree-sitter and tree-sitter-cpp.
Extract nodes of type: function_definition
For each function extract:
- function_name: the full qualified name if possible (e.g. "Parser::resolveSymbol")
  Use the declarator child node's text. Handle method definitions inside class bodies.
- file_path: relative to repo_root (use str(file_path.relative_to(repo_root)))
- start_line, end_line: 1-indexed line numbers
- code_text: full source text of the function

chunk_id = f"{file_path}::{function_name}"

Handle these file extensions: .cpp .cc .cxx .c .h .hpp .hxx
If a file fails to parse: log a warning with logger.warning(), return []
Skip empty files silently.

Write complete implementation (no stubs).
Then write tests/test_code_parser.py with pytest tests using small inline C++ strings
(not real files) to test:
- Simple function extraction
- Class method extraction  
- Multiple functions in one file
- File with parse error (returns [])
- Empty file (returns [])
```

---

## PROMPT 3 — Embedders

```
Read docs/ai_context.md.

Implement both embedding modules:

1. src/embeddings/code_embedder.py
   - Class CodeEmbedder with __init__(model_name, device)
   - Load microsoft/graphcodebert-base using AutoTokenizer and AutoModel
   - embed_chunks(chunks, batch_size=16) -> List[np.ndarray]
   - Input format: "<function> {function_name} <context> {file_path}\n{code_text}"
   - Truncate to 512 tokens max
   - Use mean pooling over last hidden state for the embedding
   - Return list of 768-dim numpy arrays

2. src/embeddings/log_embedder.py
   - Class LogEmbedder with __init__(model_name)
   - Use sentence-transformers library with all-mpnet-base-v2
   - embed_log(parsed_log) -> List[float]
   - Input: parsed_log.query_text()
   - Return 768-dim list of floats

Both classes should load the model lazily (on first embed call, not __init__)
so importing the module doesn't trigger model download.

No tests needed for embedders (they require model downloads).
But add a simple __main__ block in each file that shows a usage example.
```

---

## PROMPT 4 — Vector Index (ChromaDB)

```
Read docs/ai_context.md.

Implement src/indexing/vector_index.py completely.

Use chromadb in persistent mode.

Requirements:
- VectorIndex.__init__(persist_dir: Path) — store persist_dir, don't connect yet
- build(chunks, embeddings) — create or overwrite collection "debugaid_code_chunks"
  Store each chunk's embedding with metadata:
  {chunk_id, file_path, function_name, start_line}
  IDs = chunk.chunk_id
- query(log_embedding, top_k=20) -> List[Dict]
  Return list of: {chunk_id, file_path, function_name, start_line, score}
  score = 1 - distance (ChromaDB returns cosine distance by default)
  Raise IndexNotFoundError if collection doesn't exist
- exists() -> bool: return True if collection exists and has documents

Use chromadb.PersistentClient(path=str(persist_dir))

Write tests/test_vector_index.py using small fake embeddings (random numpy arrays)
to test build, query, and exists without real ML models.
```

---

## PROMPT 5 — BM25 Index

```
Read docs/ai_context.md.

Implement src/indexing/bm25_index.py completely.

Use rank_bm25 library (BM25Okapi class).

Requirements:
- build(chunks) — tokenize each chunk's code_text, fit BM25Okapi
  Store chunks list parallel to BM25 corpus
- query(text, top_k=20) -> List[Dict]
  Return: {chunk_id, file_path, function_name, start_line, score}
  score = raw BM25 score (float)
  Return empty list if top_k scores are all 0
- save(path: Path) — pickle both self._index and self._chunks
- load(path: Path) — unpickle and restore

_tokenize(text: str) -> List[str]:
  Split on: whitespace, (, ), {, }, ;, ,, <, >, *, &, :, ., ->, ::
  Filter out tokens shorter than 2 characters
  Filter out pure-numeric tokens
  Lowercase everything
  Keep CamelCase and snake_case identifiers intact (just lowercase them)

Write tests/test_bm25_index.py testing:
- build and query with synthetic C++ code strings
- query returns correct top result when identifier matches
- query returns empty when no match
- save and load round-trip
```

---

## PROMPT 6 — Hybrid Retriever

```
Read docs/ai_context.md and docs/2_system_architecture.md section 7.

Implement src/retrieval/hybrid_retriever.py completely.

HybridRetriever.__init__(self, vector_index: VectorIndex, bm25_index: BM25Index)

retrieve(log_embedding, log_text, top_k=5) -> List[RetrievalResult]:
  1. Get top 20 from vector_index.query(log_embedding, top_k=20)
  2. Get top 20 from bm25_index.query(log_text, top_k=20)
  3. Collect all unique chunk_ids from both result sets
  4. Normalize dense scores to [0,1]: (score - min) / (max - min + 1e-9)
  5. Normalize BM25 scores to [0,1]: same formula
  6. For each chunk_id:
     - dense_score = normalized dense score, or 0.0 if not in dense results
     - bm25_score = normalized BM25 score, or 0.0 if not in BM25 results
     - final_score = 0.6 * dense_score + 0.4 * bm25_score
  7. Sort by final_score descending
  8. Return top_k as List[RetrievalResult]

If vector_index raises IndexNotFoundError:
  log a warning, fall back to BM25-only (use bm25_score as final_score)

If bm25_index has no results:
  fall back to dense-only

Write tests/test_hybrid_retriever.py using mock VectorIndex and BM25Index
(use unittest.mock or simple fake classes) to test:
- fusion math is correct
- fallback when one index returns nothing
- results are sorted correctly
```

---

## PROMPT 7 — Evaluation Metrics

```
Read docs/ai_context.md.

Implement src/evaluation/metrics.py completely.

recall_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
  predictions = list of file_path strings, best first
  ground_truth = list of relevant file_path strings
  Return 1.0 if any ground_truth file_path appears in predictions[:k]
  Return 0.0 otherwise
  Use exact string match.

mrr_score(predictions: List[str], ground_truth: List[str]) -> float:
  Find the rank of the first prediction that appears in ground_truth
  Return 1/rank (1-indexed)
  Return 0.0 if no match found

evaluate_dataset(dataset: List[Dict], retriever, log_parser, log_embedder) -> EvalReport:
  For each item in dataset:
    parsed_log = log_parser.parse_log(item["log"])
    log_embedding = log_embedder.embed_log(parsed_log)
    results = retriever.retrieve(log_embedding, parsed_log.query_text(), top_k=5)
    predictions = [r.file_path for r in results]
    compute recall@1, recall@3, recall@5, mrr for this item
  Aggregate: mean of each metric across all items
  Also compute per error_type breakdown (group items by item["error_type"])
  Return EvalReport

Write tests/test_metrics.py with pytest testing:
- recall_at_k: hit at rank 1, hit at rank 5, miss
- mrr_score: correct rank calculation
- edge cases: empty predictions, empty ground_truth
```

---

## PROMPT 8 — CLI

```
Read docs/ai_context.md and docs/5_user_stories.md.

Implement src/cli/main.py completely using the stubs already in the file.

INDEX command:
  - Check if .debugaid/ folder exists in repo → if yes and not --force-reindex, print "Index exists. Use --force-reindex to rebuild." and exit 0
  - parse_repository(repo_path)
  - embed_chunks with CodeEmbedder
  - build VectorIndex and BM25Index, save both to repo_path/.debugaid/
  - print progress with click.echo and tqdm
  - print final: "Indexed N chunks in X seconds"

QUERY command:
  - Read log file text
  - parse_log → ParsedLog
  - Load VectorIndex and BM25Index from repo_path/.debugaid/
  - embed_log → log_embedding
  - retrieve(log_embedding, parsed_log.query_text(), top_k)
  - If output == "text": print formatted results (see README.md example)
  - If output == "json": print json.dumps(results as list of dicts)
  - Exit code 1 if no results found

EVAL command:
  - Load dataset JSON
  - Load indices from repo
  - Run evaluate_dataset
  - Print EvalReport in a clean table format

INFO command:
  - Print: repo path, number of chunks indexed, index size on disk, date built

All commands: catch exceptions, print friendly error message, exit 1.
```

---

## PROMPT 9 — Dataset Scripts

```
Read docs/4_dataset_strategy.md.

Implement scripts/mine_github_issues.py.

Use PyGithub. Read GITHUB_TOKEN from environment variable.

mine_repo(repo_name: str, max_issues: int = 200) -> List[Dict]:
  Search issues in the repo with labels ["bug"] OR body containing error keywords:
  ["undefined reference", "no such file or directory", "undeclared identifier",
   "segmentation fault", "no matching function"]
  
  For each issue:
  1. Extract log snippet: find first code block (``` fenced) in issue body
     that contains one of the error keywords
  2. Find fixing commit: search comments + linked PRs for "fixes #N" or "closes #N"
     If found, get the commit and list modified .cpp/.h files
  3. If both log snippet and relevant_files found: add to results
  
  Filter out: issues where relevant_files > 10 files, log < 20 chars

Save results to data/raw/github_issues/{repo_name_safe}.json

Main: mine llvm/llvm-project, opencv/opencv, abseil/abseil-cpp
Merge all into data/processed/github_pairs.json

Print progress and final count.
```

---

## AFTER ALL MODULES ARE DONE

Run this to verify everything connects:

```bash
# Run all tests
pytest tests/ -v

# Quick smoke test (needs a small C++ repo)
git clone --depth=1 https://github.com/abseil/abseil-cpp /tmp/abseil
debugaid index --repo /tmp/abseil
echo "undefined reference to absl::StrCat" > /tmp/test.log
debugaid query --log /tmp/test.log --repo /tmp/abseil
```


# update log parser to handle more erros

Task 1 — Expanded log parser (do this before synthetic generator)

Before building the synthetic generator, expand src/ingestion/log_parser.py to handle 3 new error categories:

asan_error — AddressSanitizer reports: patterns like heap-buffer-overflow on address, use-after-free on address, stack-buffer-overflow. Extract the error type and the stack frame where it occurred.
build_system_error — CMake/Make errors: Could not find package X, No rule to make target, undefined reference to target, CMake Error at. Extract the missing package or target name as the identifier.
runtime_exception — C++ exceptions: terminate called after throwing an instance of, std::bad_alloc, std::out_of_range, what(): . Extract the exception type and message.

Add these to ERROR_TYPES, add regex patterns to _ERROR_PATTERNS with correct priority ordering, add tests for each new category in test_log_parser.py. Update classify_error_type in scripts/mine_github_issues.py to match.