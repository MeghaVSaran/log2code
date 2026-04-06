# Symbol-Aware Retrieval Plan

## Goal

Improve DebugAid's weakest documented area: pathless debugging when logs contain symbols or stack frames but not explicit file paths.

## Planned Changes

1. Enrich parsed code chunks with lightweight symbol metadata:
   - `symbol_name`
   - `class_name`
   - `namespace`
   - `signature`
2. Feed that metadata into indexing and embedding text so retrieval sees more than raw code bodies.
3. Add a symbol-aware boost stage in hybrid retrieval:
   - exact qualified symbol matches
   - basename matches (`resolveSymbol`)
   - stack-frame matches
   - file stem matches as a weaker signal
4. Thread structured parsed-log context into retrieval from CLI and evaluation.
5. Add tests proving exact symbol matches rise above weaker generic candidates.

## Why This Slice

The docs and current-system summary agree that the real blocker is not the absence of more commands, but weak pathless retrieval. This change improves the existing `query` pipeline first, which is the foundation the later `watch`, `explain`, and `diff` modes depend on.
