# User Stories

## Story 1 — Developer With a Build Failure

**As a** software engineer working on a large C++ codebase,
**I want to** paste a build failure log into a tool and get the top 5 files responsible,
**So that** I don't spend 30 minutes manually searching through logs and code.

**Scenario:**
```
Developer runs: make -j8
Build fails with 47 lines of error output.
Developer runs: debugaid query --log build.log --repo .
Tool returns 5 files with function names and line numbers.
Developer opens the top result and finds the bug in 2 minutes.
```

**Acceptance:**
- Results returned in < 3 seconds
- Correct file appears in top 5 for 65%+ of real failures

---

## Story 2 — New Developer Who Doesn't Know the Codebase

**As a** developer who just joined a project,
**I want to** understand which part of a codebase is responsible for a failure,
**So that** I don't need senior developers to explain the code structure to me.

**Scenario:**
```
New developer clones LLVM.
Build fails immediately with a linker error.
They have no idea what "TargetRegisterInfo" is or where it lives.
They run debugaid and see:
  → llvm/lib/Target/TargetRegisterInfo.cpp [line 342]
They open that file and understand the problem.
```

**Acceptance:**
- Output includes file path, function name, and line number
- Verbose mode shows the matched code snippet

---

## Story 3 — CI Pipeline Integration

**As a** CI/CD engineer,
**I want to** automatically analyze build failures and attach debugging hints to CI reports,
**So that** developers see relevant file suggestions immediately in the PR.

**Scenario:**
```
GitHub Actions runs build → fails.
CI script runs: debugaid query --log build.log --repo . --top-k 3
Output is parsed and added as a GitHub PR comment.
Developer sees the suggestions without leaving GitHub.
```

**Acceptance:**
- CLI output is machine-parseable (supports `--output json`)
- Exit code 0 on success, 1 on no results, 2 on error

---

## Story 4 — Demo on LLVM During Interview

**As a** job candidate,
**I want to** demonstrate the tool running live on LLVM source code,
**So that** interviewers can see it work on a real million-line codebase.

**Scenario:**
```
Interviewer provides a real LLVM build failure log.
Candidate runs debugaid query --log interviewer_log.log --repo ./llvm-project
Results appear within 3 seconds.
Candidate explains: hybrid retrieval, GraphCodeBERT, evaluation metrics.
```

**Acceptance:**
- Pre-indexed LLVM repo starts up instantly (index already built)
- Works on logs the interviewer provides (not just pre-baked examples)
- Candidate can explain every design decision

---

## Story 5 — Evaluating System Quality

**As a** developer of DebugAid,
**I want to** run a benchmark against my ground truth dataset,
**So that** I know if a change I made improved or hurt retrieval quality.

**Scenario:**
```
Developer tweaks BM25 fusion weight from 0.4 to 0.5.
Runs: debugaid eval --dataset data/ground_truth/dev.json
Sees:
  Recall@1:  0.41 → 0.44  (+0.03)
  Recall@5:  0.71 → 0.73  (+0.02)
  MRR:       0.52 → 0.55  (+0.03)
  Linker errors:   0.82 Recall@5
  Compiler errors: 0.69 Recall@5
  Include errors:  0.74 Recall@5
Developer commits the change.
```

**Acceptance:**
- Eval script outputs per-category breakdown
- Runs in < 5 minutes on 500-sample dataset
