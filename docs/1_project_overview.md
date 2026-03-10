# DebugAid — ML-Assisted Log–Code Correlation Engine

## Problem

When a C++ build or runtime failure occurs, engineers manually search through
thousands of log lines to find the responsible file and function. This is slow,
experience-dependent, and does not scale to large codebases like LLVM or Chromium.

## Solution

DebugAid automatically maps error logs to the most relevant source files and
functions using semantic embeddings and hybrid retrieval. It works on any C++
codebase without modification.

## Core Idea

- Logs and code are converted into vector representations
- A hybrid retrieval system (dense semantic search + BM25 keyword search) finds
  the most relevant code given a log
- Results are ranked and returned with file paths, function names, and line numbers

## Primary Use Case

```
$ debugaid --log build.log --repo ./llvm-project

Top matches for: "undefined reference to Parser::resolveSymbol"

1. src/parser/resolve.cpp       — Parser::resolveSymbol()     [line 142]  score: 0.91
2. include/parser/symbol_table.h — SymbolTable::lookup()      [line 67]   score: 0.84
3. src/compiler/analyze.cpp     — Compiler::analyze()         [line 203]  score: 0.79
4. src/parser/parse_expr.cpp    — Parser::parseExpression()   [line 88]   score: 0.71
5. src/linker/link_resolver.cpp — Linker::resolveAll()        [line 315]  score: 0.68
```

## What Makes This Different From Grep / String Search

- Works even when log messages do not contain exact function names
- Handles mangled C++ symbols, template errors, and segfault stack traces
- Semantic understanding: a log about "missing symbol during linking" maps to
  linker and symbol table code even without keyword overlap
- Scales to million-line codebases

## Tech Stack

| Layer            | Tool                              |
|------------------|-----------------------------------|
| Language         | Python 3.11+                      |
| Code parsing     | Tree-sitter                       |
| Code embeddings  | GraphCodeBERT                     |
| Log embeddings   | all-mpnet-base-v2                 |
| Vector store     | ChromaDB                          |
| Sparse retrieval | rank_bm25                         |
| Evaluation       | Custom (Recall@K, MRR)            |
| Interface        | CLI (Click)                       |
| Testing          | pytest                            |

## Target Codebases for Demo

- LLVM / Clang (primary demo target)
- Linux kernel
- OpenCV
- Any C++ project with a CMake/Make build system

## Success Criteria

- Recall@5 > 0.70 on ground truth dataset
- MRR > 0.50 on ground truth dataset
- Indexes a 500k-line C++ codebase in under 10 minutes
- Returns results in under 3 seconds per query
