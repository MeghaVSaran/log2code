# Log Parser Implementation

Implement the 4 stub functions in [src/ingestion/log_parser.py](file:///c:/projects/log2code/src/ingestion/log_parser.py) and create complete `tests/test_log_parser.py`.

## Proposed Changes

### Log Parser

#### [MODIFY] [log_parser.py](file:///c:/projects/log2code/src/ingestion/log_parser.py)

The existing file has the [ParsedLog](file:///c:/projects/log2code/src/ingestion/log_parser.py#27-41) dataclass and 4 stub functions. I will keep the dataclass and docstrings untouched and implement each function:

**1. [extract_error_type(log_text) -> str](file:///c:/projects/log2code/src/ingestion/log_parser.py#57-68)**
- Priority-ordered regex matching (most specific first):
  1. `include_error` — `fatal error:.*No such file or directory`
  2. `template_error` — `implicit instantiation of undefined template` or `undefined template`
  3. `linker_error` — `undefined reference to` or `multiple definition of`
  4. `compiler_error` — `use of undeclared identifier` or `no matching function for call to`
  5. `segfault` — `Segmentation fault` or stack frames `#\d+\s+\S+::\S+`
- If none match → `"unknown"`
- Ordering rationale: include/template errors are more specific than generic compiler errors. Linker before compiler because `undefined reference` is linker-specific.

**2. [extract_identifiers(log_text) -> List[str]](file:///c:/projects/log2code/src/ingestion/log_parser.py#70-85)**
- Patterns extracted:
  - `undefined reference to '?(\S+)'?` → symbol name (strip backticks/quotes)
  - `use of undeclared identifier '?(\S+)'?`
  - `multiple definition of '?(\S+)'?`
  - `no matching function for call to '?(\S+)'?`
  - Stack frames: `#\d+\s+(?:0x[\da-f]+\s+in\s+)?(\S+::\S+)` → class::method
- Post-processing: strip C++ mangling suffixes (trailing [(...)](file:///c:/projects/log2code/src/ingestion/log_parser.py#43-55) parameter lists), deduplicate, preserve order.

**3. [extract_file_hints(log_text) -> List[str]](file:///c:/projects/log2code/src/ingestion/log_parser.py#87-97)**
- Match filenames with C++ extensions: `\b([\w./\\-]+\.(?:cpp|h|hpp|cc|cxx|c|hxx))\b`
- Also catch the include-error pattern: `fatal error: (\S+\.h): No such file`
- Deduplicate and return.

**4. [parse_log(log_text) -> ParsedLog](file:///c:/projects/log2code/src/ingestion/log_parser.py#43-55)**
- Orchestrates the above 3 functions.
- Picks `error_message`: the single most informative line. Scans lines for the one matching the detected error type pattern; falls back to first non-empty line.
- Extracts `stack_frames` separately: lines matching `#\d+\s+` pattern.
- Assembles and returns [ParsedLog](file:///c:/projects/log2code/src/ingestion/log_parser.py#27-41).

---

### Tests

#### [NEW] [test_log_parser.py](file:///c:/projects/log2code/tests/test_log_parser.py)

8 test functions using `pytest`:

| Test | What it covers |
|------|---------------|
| `test_linker_error` | `undefined reference to` — verifies error_type, identifier extraction, query_text |
| `test_compiler_error` | `use of undeclared identifier` |
| `test_include_error` | `fatal error: X.h: No such file or directory` — also checks file_hints |
| `test_template_error` | `implicit instantiation of undefined template` |
| `test_segfault` | `Segmentation fault` with stack frames — checks stack_frames field |
| `test_multiline_log` | Multi-line log with noise lines around the error |
| `test_unknown_error` | Random text with no known pattern → `error_type="unknown"` |
| `test_multiple_error_lines` | Log containing both include and compiler errors → should return the most specific (`include_error`) |

## Verification Plan

### Automated Tests

Run from project root with:

```
python -m pytest tests/test_log_parser.py -v
```

All 8 tests should pass. No external dependencies needed (pure regex, no models).
