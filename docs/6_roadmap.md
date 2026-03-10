# Development Roadmap — MVP

## Overview

4-week plan to a working, evaluated MVP.
Each week has a clear deliverable you can show a supervisor.

---

## Week 1 — Ingestion + Parsing

**Goal:** Given a C++ repo, produce structured chunks. Given a log, produce structured error info.

### Tasks

- [ ] Set up repo structure, virtual env, requirements.txt
- [ ] Implement `src/ingestion/code_parser.py`
  - Tree-sitter walks repo
  - Extracts function-level chunks
  - Handles all C++ extensions
  - Skips parse failures gracefully
- [ ] Implement `src/ingestion/log_parser.py`
  - 5 error category parsers (linker, compiler, include, template, segfault)
  - Returns ParsedLog dataclass
- [ ] Write tests: `tests/test_code_parser.py`, `tests/test_log_parser.py`
- [ ] Manually verify on 3 real LLVM log examples

### Deliverable

```
python -c "
from src.ingestion.code_parser import parse_repository
chunks = parse_repository('./llvm-project/clang/lib/Parse')
print(f'Extracted {len(chunks)} chunks')
print(chunks[0])
"
```

---

## Week 2 — Embeddings + Indexing

**Goal:** Chunks and logs are converted to vectors. Indices are built and queryable.

### Tasks

- [ ] Implement `src/embeddings/code_embedder.py`
  - GraphCodeBERT, batched, handles long inputs
- [ ] Implement `src/embeddings/log_embedder.py`
  - all-mpnet-base-v2
- [ ] Implement `src/indexing/vector_index.py`
  - ChromaDB persistent index
  - build / query / load / exists
- [ ] Implement `src/indexing/bm25_index.py`
  - rank_bm25 wrapper
  - build / query / save / load
- [ ] Write tests for all indexing modules
- [ ] Benchmark: how long to index 50k lines of LLVM?

### Deliverable

```
python scripts/build_index.py --repo ./llvm-project/clang/lib/Parse
# prints: Indexed N chunks in X seconds
```

---

## Week 3 — Retrieval + CLI + Evaluation

**Goal:** End-to-end pipeline works. CLI is usable. Metrics are computed.

### Tasks

- [ ] Implement `src/retrieval/hybrid_retriever.py`
  - Dense + BM25 fusion
  - Returns RetrievalResult list
- [ ] Implement `src/evaluation/metrics.py`
  - Recall@1, Recall@3, Recall@5
  - MRR
- [ ] Implement `src/cli/main.py`
  - `index`, `query`, `eval`, `info` commands
- [ ] Write `tests/test_hybrid_retriever.py`
- [ ] Test manually: run `debugaid query --log examples/linker_error.log --repo ./llvm-project`

### Deliverable

Working CLI that returns results for a real log.

---

## Week 4 — Dataset + Evaluation + Polish

**Goal:** Measured Recall@5 and MRR on real ground truth data. README done.

### Tasks

- [ ] Implement `scripts/mine_github_issues.py`
  - Mine LLVM, OpenCV GitHub issues
  - Store 200+ labeled pairs
- [ ] Implement `scripts/generate_synthetic_errors.py`
  - Generate 300+ synthetic pairs from LLVM source
- [ ] Merge, clean, deduplicate → `data/ground_truth/dataset.json`
- [ ] Run `debugaid eval --dataset data/ground_truth/dev.json`
- [ ] Tune BM25 fusion weight if Recall@5 < 0.60
- [ ] Write README.md (project description, install, usage, metrics)
- [ ] Prepare LLVM demo: pre-index full llvm-project, have 5 demo logs ready

### Deliverable

```
debugaid eval --dataset data/ground_truth/dev.json

Results on 200 samples:
  Recall@1:  0.38
  Recall@3:  0.59
  Recall@5:  0.71
  MRR:       0.51

By error type:
  linker_error:    Recall@5 = 0.82
  compiler_error:  Recall@5 = 0.66
  include_error:   Recall@5 = 0.74
  segfault:        Recall@5 = 0.61
```

---

## Phase 2 (Post-MVP)

After supervisor review and testing on full LLVM:

- Cross-encoder re-ranking (Week 5–6)
- UMAP + HDBSCAN error clustering (Week 7)
- LLM explanation layer via API (Week 7–8)
- Streamlit UI (Week 8)
- Incremental index updates

---

## Daily Workflow

```
1. Pick one task from the current week
2. Open docs/ai_context.md
3. Write a specific prompt for Claude Code / Codex
4. Review generated code against architecture docs
5. Run tests: pytest tests/
6. Commit if tests pass
```

## Git Commit Convention

```
feat: implement log_parser for linker errors
fix: handle template function chunks in code_parser
test: add recall@k tests for edge cases
eval: run benchmark on 200 dev samples
docs: update architecture with BM25 fusion details
```
