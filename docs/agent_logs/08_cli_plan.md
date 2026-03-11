# CLI — Implementation Plan

## Goal
Implement `src/cli/main.py` with 4 click commands: INDEX, QUERY, EVAL, INFO.

## Changes

### [MODIFY] main.py (was stub)
- `index`: check .debugaid/, parse_repository, embed, build vector+BM25, save
- `query`: read log, parse, load indices, embed, retrieve, format text/json output
- `eval`: load dataset JSON, load indices, run evaluate_dataset, print table
- `info`: show repo path, chunk count, index size, date built
- All: lazy imports, try/except with friendly error messages, exit codes
