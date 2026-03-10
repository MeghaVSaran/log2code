# Dataset Strategy

## Goal

Build a ground truth dataset of (log_snippet, relevant_files[]) pairs
sufficient to evaluate the retrieval system with Recall@K and MRR.

Target: 500–2000 labeled pairs covering all 5 error categories.

---

## Source 1 — GitHub Issue Mining (Primary)

### Strategy

Mine public C++ repositories for issues that contain build/runtime logs
and have linked fixing commits.

### Target Repositories

| Repository | Why |
|---|---|
| llvm/llvm-project | Large, well-maintained, many CI failures |
| gcc-mirror/gcc | Classic compiler, linker errors |
| opencv/opencv | Popular, many contributor issues |
| abseil/abseil-cpp | Modern C++, good issue hygiene |
| google/protobuf | Build system diversity |

### Mining Process

```
1. GitHub API: search issues with label "bug" or body containing:
   - "undefined reference"
   - "no such file or directory"
   - "use of undeclared identifier"
   - "segmentation fault"
   - "error: no matching function"

2. For each issue:
   a. Extract log snippet from issue body (first code block containing error)
   b. Find linked PR or commit (look for "fixes #N", "closes #N" in commits)
   c. Extract list of modified .cpp/.h files from the commit diff
   d. Store: {log, relevant_files, repo, issue_url, commit_sha}

3. Filter:
   - Skip if no code block found in issue
   - Skip if fixing commit modifies > 10 files (too noisy)
   - Skip if log snippet < 20 characters
```

### Output Format

```json
{
  "id": "llvm-issue-12345",
  "log": "undefined reference to `llvm::TargetRegisterInfo::getRegAsmName`",
  "relevant_files": [
    "llvm/lib/Target/TargetRegisterInfo.cpp",
    "llvm/include/llvm/Target/TargetRegisterInfo.h"
  ],
  "error_type": "linker_error",
  "source": "github",
  "repo": "llvm/llvm-project",
  "issue_url": "https://github.com/llvm/llvm-project/issues/12345"
}
```

### Script

`scripts/mine_github_issues.py`

Requirements: `PyGithub`, `GITHUB_TOKEN` env variable

Expected yield: 300–600 labeled pairs across all repos.

---

## Source 2 — Synthetic Error Generation (Gap Filler)

### Strategy

Programmatically introduce common C++ errors into real source files,
compile, capture the log. Ground truth = the file that was modified.

### Error Types to Synthesize

| Error Type | Injection Method | Expected Log Pattern |
|---|---|---|
| Undefined reference | Comment out function body, keep declaration | `undefined reference to X` |
| Multiple definition | Copy function definition to second file | `multiple definition of X` |
| Missing include | Remove `#include` line | `fatal error: X.h: No such file` |
| Undeclared identifier | Rename a function at call site only | `use of undeclared identifier X` |
| Type mismatch | Change argument type at call site | `no matching function for call to` |

### Process

```python
# Pseudocode for synthetic generator
for source_file in sample_cpp_files(repo, n=200):
    for error_type in ERROR_TYPES:
        modified_file, expected_log_pattern = inject_error(source_file, error_type)
        log = compile_and_capture(modified_file, repo_build_config)
        if log matches expected_log_pattern:
            save_sample({
                "log": extract_error_lines(log),
                "relevant_files": [source_file],
                "error_type": error_type,
                "source": "synthetic"
            })
        restore_file(source_file)
```

### Compile Setup

Use a subset of LLVM or OpenCV that has a working CMake build.
Inject errors one at a time. Capture stderr.

### Script

`scripts/generate_synthetic_errors.py`

Expected yield: 500–1500 labeled pairs.

---

## Source 3 — LogHub (For Clustering Evaluation Only)

**URL:** https://github.com/logpai/loghub

LogHub contains labeled system logs from 30+ real systems.
Not directly useful for log→code mapping, but useful for:

- Validating error type classification
- Testing log parser on diverse log formats
- Phase 2: clustering evaluation

---

## Dataset Splits

```
Total target: 800–2000 pairs

Train / Dev split: 80% / 20%
  - Train: used during development to tune fusion weights
  - Dev: used for all reported metrics

No test set needed for MVP (system is retrieval-based, not trained end-to-end).
```

---

## Dataset Storage

```
data/
  raw/
    github_issues/          ← raw API responses
    synthetic_logs/         ← raw compile outputs
  processed/
    github_pairs.json       ← cleaned github pairs
    synthetic_pairs.json    ← cleaned synthetic pairs
  ground_truth/
    dataset.json            ← merged, deduplicated, split dataset
    train.json
    dev.json
```

### dataset.json schema

```json
[
  {
    "id": "string",
    "log": "string",
    "relevant_files": ["string"],
    "error_type": "linker_error | compiler_error | include_error | template_error | segfault",
    "source": "github | synthetic",
    "split": "train | dev"
  }
]
```

---

## Quality Checks

Before including a sample, verify:

1. `log` field is non-empty and contains a recognizable error pattern
2. `relevant_files` has at least 1 entry
3. All files in `relevant_files` actually exist in the target repository
4. Log does not contain only a file path with no error message
5. Duplicates removed by hashing `(log[:100], relevant_files[0])`

---

## Expected Dataset Distribution by Error Type

| Error Type | GitHub | Synthetic | Total |
|---|---|---|---|
| Linker error | 150 | 300 | 450 |
| Compiler error | 100 | 300 | 400 |
| Include error | 80 | 200 | 280 |
| Template error | 50 | 100 | 150 |
| Segfault | 70 | 100 | 170 |
| **Total** | **450** | **1000** | **1450** |
