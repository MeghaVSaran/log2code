# CLI — Walkthrough

## Changes Made

### main.py (was stub → fully implemented)
- 4 click commands: `index`, `query`, `eval`, `info`
- Lazy imports inside each command for fast CLI startup
- text/json output formats for query
- Verbose mode shows scores
- Per-error-type eval table
- Friendly error messages with exit code 1

## Verification

- CLI help: ✅ all commands listed
- Index help: ✅ --repo and --force-reindex shown
- All 106 existing tests pass ✅ (5.73s)
- No CLI-specific tests (commands require real models/repos)
