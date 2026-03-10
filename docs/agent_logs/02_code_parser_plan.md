# Code Parser — Implementation Plan

## Goal
Implement `src/ingestion/code_parser.py` using tree-sitter 0.25 to extract function-level chunks from C++ repositories.

## Proposed Changes

### [MODIFY] code_parser.py
- `parse_file()`: Read file → parse with tree-sitter → walk AST for `function_definition` nodes → build Chunks
- `parse_repository()`: rglob all CPP_EXTENSIONS → call parse_file on each → aggregate
- Handle `template_declaration` by descending into child `function_definition` nodes
- `_find_enclosing_class()`: walk parent chain for `class_specifier` to qualify inline method names
- `_extract_function_name()`: 3 cases — qualified_identifier, field_identifier + class, plain identifier
- Windows path normalization: `.replace("\\", "/")`

### [NEW] test_code_parser.py
8 test classes using `tmp_path` with inline C++ strings:
1. Simple function
2. Class method (out-of-class)
3. Multiple functions
4. Parse error (binary gibberish)
5. Empty file
6. Template function
7. Header inline method (.hpp)
8. parse_repository integration
