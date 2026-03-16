# Implement `scripts/mine_github_issues.py`

GitHub issue mining script to build a labeled dataset of (log_snippet, relevant_files) pairs by mining bug issues from public C++ repositories.

## Proposed Changes

### Dataset Scripts

#### [NEW] [mine_github_issues.py](file:///home/vsant/Megha/log2code/scripts/mine_github_issues.py)

**Purpose**: Mine GitHub issues from C++ repos for labeled error log → source file pairs.

**Key functions**:

- **`mine_repo(repo_name: str, max_issues: int = 200) -> List[Dict]`**
  - Uses PyGithub with `GITHUB_TOKEN` env var
  - **Prints `g.get_rate_limit().core.remaining` at the start of each repo**
  - Searches issues with `label:bug` OR body containing error keywords: `"undefined reference"`, `"no such file or directory"`, `"undeclared identifier"`, `"segmentation fault"`, `"no matching function"`
  - **Rate limit handling**: catches `github.GithubException` with status 403 or 429, sleeps 60 seconds and retries (up to 3 retries per request)
  - For each issue:
    1. Extracts log snippet: finds first ` ``` `-fenced code block in issue body that contains one of the error keywords
    2. Finds fixing commit: scans comments + linked PRs for `"fixes #N"` or `"closes #N"` patterns; if found, retrieves commit and lists modified `.cpp`/`.h` files
    3. If both log snippet and relevant_files found → adds to results
    4. **Saves results incrementally** to `data/raw/github_issues/{repo_name_safe}.json` after each successful issue (crash-safe)
  - Filters: skips issues where `relevant_files > 10` or `log < 20 chars`
  - On startup, loads any existing partial results from the JSON file so interrupted runs can resume

- **`classify_error_type(log_text: str) -> str`** — lightweight classifier reusing patterns from `src/ingestion/log_parser.py`

- **`main()`** block:
  - Mines: `llvm/llvm-project`, `opencv/opencv`, `abseil/abseil-cpp`
  - Merges all results into `data/processed/github_pairs.json`
  - Prints progress per-repo and final count

**Output schema** (per `docs/4_dataset_strategy.md`):
```json
{
  "id": "llvm-issue-12345",
  "log": "undefined reference to ...",
  "relevant_files": ["path/to/file.cpp"],
  "error_type": "linker_error",
  "source": "github",
  "repo": "llvm/llvm-project",
  "issue_url": "https://github.com/llvm/llvm-project/issues/12345"
}
```

**Dependencies**: `PyGithub` (already in `requirements.txt`), `pathlib`, `json`, `re`, `os`, `logging`

---

### Directory Structure

Creates the following directories (via `pathlib.Path.mkdir(parents=True, exist_ok=True)`):
- `data/raw/github_issues/`
- `data/processed/`

No existing files are modified.

## Verification Plan

### Automated Tests

Since this script depends on GitHub API access (requires `GITHUB_TOKEN`), it is not suitable for unit testing without mocking. Verification will be:

1. **Syntax check**: `python -c "import ast; ast.parse(open('scripts/mine_github_issues.py').read())"` to ensure no syntax errors.

2. **Module import check**: `python -c "import scripts.mine_github_issues"` — verify the script imports cleanly.

3. **Existing test suite**: `pytest tests/ -v` — ensure nothing in the existing codebase is broken.

### Manual Verification

- Script requires a valid `GITHUB_TOKEN` environment variable to actually execute, which the user can test by running:
  ```bash
  export GITHUB_TOKEN=ghp_your_token_here
  python scripts/mine_github_issues.py
  ```
