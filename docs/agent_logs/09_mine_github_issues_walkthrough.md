# Walkthrough: `scripts/mine_github_issues.py`

## What Was Built

Created `scripts/mine_github_issues.py` — a GitHub issue mining script that produces labeled `(log_snippet, relevant_files[])` pairs for the DebugAid evaluation dataset.

## Files Created

| File | Purpose |
|------|---------|
| `scripts/__init__.py` | Package init for scripts module |
| `scripts/mine_github_issues.py` | Main mining script (491 lines) |
| `docs/agent_logs/09_mine_github_issues_plan.md` | Implementation plan |

## Key Functions

### `mine_repo(repo_name, max_issues=200) -> List[Dict]`
- Connects to GitHub via `GITHUB_TOKEN` env var
- Prints rate limit quota at start of each repo
- Searches issues with `label:bug` + keyword body search
- Extracts log snippets from fenced code blocks
- Finds fixing commits via issue events, comments, and commit messages
- Saves incrementally after every successful issue (crash-safe)
- Resumes from existing partial results on restart

### `classify_error_type(log_text) -> str`
- Mirrors `src/ingestion/log_parser.py` patterns (standalone, no internal imports)
- Covers: include_error, template_error, linker_error, compiler_error, segfault

### `_find_fixing_commit(repo, issue) -> Optional[List[str]]`
Four-stage fixing commit detection:
1. Scan comments for commit SHA references
2. Use issue events API (`closed`/`referenced` events with `commit_id`)
3. Check if issue is itself a merged PR
4. Bounded commit message search (last 200 commits)

### `_api_call_with_retry(func, *args, **kwargs)`
- Catches HTTP 403/429 `GithubException` (rate limit)
- Sleeps 60s and retries up to 3 times

### `_save_results(results, path)` / `_load_existing_results(path)`
- Atomic write via `.tmp` → `.replace()` pattern
- Resume support: loads existing JSON on startup, tracks `seen_ids`

## Output

- Per-repo raw: `data/raw/github_issues/{repo_name_safe}.json`
- Merged: `data/processed/github_pairs.json`
- Schema matches `docs/4_dataset_strategy.md`

## Verification

- **Code review**: All imports valid, type hints on all signatures, Google-style docstrings, no circular imports
- **Terminal verification blocked** by workspace path validation — user should run:
  ```bash
  cd /home/vsant/Megha/log2code
  source venv/bin/activate
  python -c "import ast; ast.parse(open('scripts/mine_github_issues.py').read()); print('OK')"
  ```
- **Live run** requires `GITHUB_TOKEN`:
  ```bash
  export GITHUB_TOKEN=ghp_your_token
  python scripts/mine_github_issues.py
  ```

## Build Order

Build order: file 9 of ~17 total project files
