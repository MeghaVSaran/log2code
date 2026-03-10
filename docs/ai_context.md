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
log2code/
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

---

## Git Commit Requirements

After completing any implementation task, always generate a ready-to-run
git commit command with an elaborated commit message. Never commit yourself —
just output the command for the developer to review and run.

### Commit Format

Use this exact structure:
```
git add <specific files changed, never use git add .>
git commit -m "<type>(<scope>): <short summary>

<what was implemented, 3-8 bullet points>
- 
- 
-

<why these decisions were made>

<any edge cases handled or known limitations>"
```

### Commit Types

- feat     → new implementation (filling a stub)
- fix      → bug fix in existing implementation  
- test     → adding or fixing tests
- refactor → restructuring without behavior change
- docs     → documentation changes

### Scope = the module name

Examples: log_parser, code_parser, hybrid_retriever, bm25_index

### Example of a Good Commit Output
```
git add src/ingestion/log_parser.py tests/test_log_parser.py
git commit -m "feat(log_parser): implement regex-based C++ error log parser

- Implement extract_error_type() with priority-ordered regex matching
  covering 5 categories: include, template, linker, compiler, segfault
- Implement extract_identifiers() stripping C++ mangled parameter lists
  and deduplicating across all error patterns
- Implement extract_file_hints() matching all C++ file extensions
- Implement parse_log() orchestrating above functions into ParsedLog
- Handle multi-line logs by selecting most informative error line
- Handle namespace-qualified identifiers (e.g. llvm::SelectionDAG::getNode)

Priority ordering (most specific first) prevents misclassification when
a log contains both include errors and compiler errors simultaneously.

Edge cases handled:
- Logs with no recognizable pattern return error_type='unknown'
- Mangled symbols with parameter lists are stripped to bare function name
- Stack frames in both GDB and LLVM format are parsed correctly"
```

### Rules

- Always list specific files in git add, never wildcard
- Always include what AND why in the message body
- Always mention edge cases that were explicitly handled
- If multiple files were changed, explain each file's changes separately

---

## File Creation Requirements

When creating any new file, always follow these rules:

### Rule 1 — Always State the Full Path Before Writing

Before writing any file, explicitly state:
"Creating file at: src/ingestion/log_parser.py"

Never assume the root directory. Always use the full path
relative to the project root as defined in the Repository
Structure section of this document.

### Rule 2 — File Path Reference Table

When creating files, always place them here:

| File type                  | Correct path                        |
|----------------------------|-------------------------------------|
| C++ ingestion modules      | src/ingestion/                      |
| Embedding modules          | src/embeddings/                     |
| Index modules              | src/indexing/                       |
| Retrieval modules          | src/retrieval/                      |
| Evaluation modules         | src/evaluation/                     |
| CLI modules                | src/cli/                            |
| Dataset scripts            | scripts/                            |
| Test files                 | tests/                              |
| Ground truth data          | data/ground_truth/                  |
| Raw scraped data           | data/raw/                           |
| Documentation              | docs/                               |
| Config files               | project root only (requirements.txt,|
|                            | setup.py, .gitignore, README.md)    |

### Rule 3 — New File Commit Message Format

When a file is being created for the first time (not modifying
an existing stub), use this extended commit format so the git
history clearly shows what was created, where, and why:
```
git add <full/path/to/new_file.py>
git commit -m "feat(<scope>): create <filename> at <full/path/>

NEW FILE: <full/path/to/new_file.py>
Purpose: <one sentence on what this file does>

Contents:
- <class or function 1>: <what it does>
- <class or function 2>: <what it does>
- <class or function 3>: <what it does>

Placed in <directory/> because <reason — e.g. 'this is the
ingestion layer responsible for all input parsing'>.

Dependencies introduced:
- <library>: <why it was needed>

Next file that depends on this: <module that will import this>"
```

### Rule 4 — When Modifying an Existing Stub

When filling in a stub file that already exists, use this format
to make clear in history that this was an implementation pass,
not a new file:
```
git add <full/path/to/file.py>
git commit -m "feat(<scope>): implement <filename> (was stub)

IMPLEMENTS STUB: <full/path/to/file.py>
Previously: empty stub with NotImplementedError
Now: fully implemented

Changes:
- <function 1>: implemented with <approach>
- <function 2>: implemented with <approach>

Edge cases handled:
- <case 1>
- <case 2>"
```

### Rule 5 — Ordering Annotation in Commit Body

Every commit message body must include this line:
```
Build order: file N of ~17 total project files
```

Where N reflects the sequence in which files are being built
(log_parser is ~1, code_parser ~2, code_embedder ~3, etc.)
so the git log tells the full story of how the project was
assembled in order.

### Example of a Complete New File Commit
```
git add src/ingestion/log_parser.py tests/test_log_parser.py
git commit -m "feat(log_parser): create log_parser.py at src/ingestion/

NEW FILE: src/ingestion/log_parser.py
Purpose: regex-based parser that converts raw C++ error logs into
structured ParsedLog objects for downstream embedding and retrieval.

Contents:
- ParsedLog dataclass: structured container for all extracted fields
- parse_log(): orchestrates extraction into a single ParsedLog
- extract_error_type(): priority-ordered regex classification into
  5 categories (include, template, linker, compiler, segfault)
- extract_identifiers(): pulls C++ symbol names, strips mangling
- extract_file_hints(): pulls filenames with C++ extensions
- query_text(): formats log for use as embedder input

Placed in src/ingestion/ because this is the ingestion layer —
all raw input parsing (both logs and code) lives here.

Dependencies introduced:
- re (stdlib): regex pattern matching
- dataclasses (stdlib): ParsedLog structure
- No external dependencies — this module is pure Python

Next file that depends on this: src/embeddings/log_embedder.py
which calls parsed_log.query_text() as its input.

NEW FILE: tests/test_log_parser.py
Purpose: pytest suite covering all 5 error categories and edge cases.

Build order: file 1 of ~17 total project files"
```

---

## Agent Artifact Requirements

After every implementation task, two additional files must be saved
in addition to the source code:

### 1. Implementation Plan
Save the plan the agent generated BEFORE writing code.

Path: docs/agent_logs/{NN}_{module_name}_plan.md
Where NN = zero-padded build order number (01, 02, 03...)

Example: docs/agent_logs/01_log_parser_plan.md

### 2. Walkthrough
Save the post-implementation walkthrough with the diff and
verification results.

Path: docs/agent_logs/{NN}_{module_name}_walkthrough.md

Example: docs/agent_logs/01_log_parser_walkthrough.md

### Commit for Agent Artifacts

Always include these files in the same commit as the source code:
```
git add src/ingestion/log_parser.py
git add tests/test_log_parser.py
git add docs/agent_logs/01_log_parser_plan.md
git add docs/agent_logs/01_log_parser_walkthrough.md
git commit -m "feat(log_parser): implement regex-based C++ error log parser

..."
```

Never commit source code without its corresponding plan and walkthrough.
This ensures the git history explains not just WHAT changed but WHY
each decision was made.
