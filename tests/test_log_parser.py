"""
Tests for src/ingestion/log_parser.py

Covers all 5 error categories, multi-line logs, unknown errors,
multiple-error-line priority, and deeply-namespaced identifiers.
"""

import pytest
from src.ingestion.log_parser import (
    ParsedLog,
    parse_log,
    extract_error_type,
    extract_identifiers,
    extract_file_hints,
)


# ---------------------------------------------------------------------------
# 1. One test per error category
# ---------------------------------------------------------------------------

class TestLinkerError:
    """Linker error: 'undefined reference to' and 'multiple definition of'."""

    LOG_UNDEF_REF = (
        "/usr/bin/ld: main.o: in function `main':\n"
        "main.cpp:(.text+0x1a): undefined reference to `Parser::resolveSymbol'\n"
        "collect2: error: ld returned 1 exit status\n"
    )

    LOG_MULTI_DEF = (
        "/usr/bin/ld: symbol_table.o: multiple definition of `SymbolTable::insert';\n"
        "  first defined here\n"
        "collect2: error: ld returned 1 exit status\n"
    )

    def test_undef_ref_error_type(self):
        result = parse_log(self.LOG_UNDEF_REF)
        assert result.error_type == "linker_error"

    def test_undef_ref_identifiers(self):
        result = parse_log(self.LOG_UNDEF_REF)
        assert "Parser::resolveSymbol" in result.identifiers

    def test_undef_ref_query_text(self):
        result = parse_log(self.LOG_UNDEF_REF)
        assert "Parser::resolveSymbol" in result.query_text()

    def test_multi_def_error_type(self):
        result = parse_log(self.LOG_MULTI_DEF)
        assert result.error_type == "linker_error"

    def test_multi_def_identifiers(self):
        result = parse_log(self.LOG_MULTI_DEF)
        assert "SymbolTable::insert" in result.identifiers


class TestCompilerError:
    """Compiler error: 'use of undeclared identifier' and 'no matching function'."""

    LOG_UNDECLARED = (
        "main.cpp:42:10: error: use of undeclared identifier 'resolveSymbol'\n"
        "    resolveSymbol(sym);\n"
        "    ^\n"
    )

    LOG_NO_MATCH = (
        "parser.cpp:88:5: error: no matching function for call to 'Parser::parse'\n"
        "    parser.parse(input, flags);\n"
        "           ^~~~~\n"
    )

    def test_undeclared_error_type(self):
        result = parse_log(self.LOG_UNDECLARED)
        assert result.error_type == "compiler_error"

    def test_undeclared_identifiers(self):
        result = parse_log(self.LOG_UNDECLARED)
        assert "resolveSymbol" in result.identifiers

    def test_no_match_error_type(self):
        result = parse_log(self.LOG_NO_MATCH)
        assert result.error_type == "compiler_error"

    def test_no_match_identifiers(self):
        result = parse_log(self.LOG_NO_MATCH)
        assert "Parser::parse" in result.identifiers

    def test_file_hints(self):
        result = parse_log(self.LOG_UNDECLARED)
        assert "main.cpp" in result.file_hints


class TestIncludeError:
    """Include error: 'fatal error: X.h: No such file or directory'."""

    LOG = (
        "In file included from main.cpp:1:\n"
        "fatal error: parser/resolve.h: No such file or directory\n"
        "    #include <parser/resolve.h>\n"
        "             ^~~~~~~~~~~~~~~~~~\n"
        "compilation terminated.\n"
    )

    def test_error_type(self):
        result = parse_log(self.LOG)
        assert result.error_type == "include_error"

    def test_file_hints(self):
        result = parse_log(self.LOG)
        assert "parser/resolve.h" in result.file_hints

    def test_error_message_contains_filename(self):
        result = parse_log(self.LOG)
        assert "resolve.h" in result.error_message


class TestTemplateError:
    """Template error: 'implicit instantiation of undefined template'."""

    LOG = (
        "vector_ops.cpp:15:3: error: implicit instantiation of undefined template "
        "'std::vector<CustomType>'\n"
        "  std::vector<CustomType> v;\n"
        "  ^\n"
    )

    def test_error_type(self):
        result = parse_log(self.LOG)
        assert result.error_type == "template_error"

    def test_error_message(self):
        result = parse_log(self.LOG)
        assert "implicit instantiation" in result.error_message


class TestSegfault:
    """Segfault: 'Segmentation fault' with stack frames."""

    LOG = (
        "Program received signal SIGSEGV, Segmentation fault.\n"
        "#0  0x00005555555551a9 in Parser::resolveSymbol (this=0x0, s=...) at resolve.cpp:42\n"
        "#1  0x0000555555555230 in SymbolTable::lookup (this=0x7fffffffde10) at symbol_table.cpp:88\n"
        "#2  0x00005555555552f0 in main () at main.cpp:15\n"
    )

    def test_error_type(self):
        result = parse_log(self.LOG)
        assert result.error_type == "segfault"

    def test_stack_frames_extracted(self):
        result = parse_log(self.LOG)
        assert len(result.stack_frames) == 3

    def test_identifiers_from_frames(self):
        result = parse_log(self.LOG)
        assert "Parser::resolveSymbol" in result.identifiers
        assert "SymbolTable::lookup" in result.identifiers

    def test_file_hints_from_frames(self):
        result = parse_log(self.LOG)
        assert "resolve.cpp" in result.file_hints
        assert "symbol_table.cpp" in result.file_hints


# ---------------------------------------------------------------------------
# 2. Multi-line log handling
# ---------------------------------------------------------------------------

class TestMultilineLog:
    """Multi-line log with noise lines surrounding the actual error."""

    LOG = (
        "make[2]: Entering directory '/home/user/project/build'\n"
        "g++ -c -o main.o main.cpp\n"
        "In file included from utils.h:3:\n"
        "main.cpp:42:10: error: use of undeclared identifier 'processData'\n"
        "    processData(buffer);\n"
        "    ^\n"
        "1 error generated.\n"
        "make[2]: *** [Makefile:15: main.o] Error 1\n"
        "make[1]: Leaving directory '/home/user/project/build'\n"
    )

    def test_error_type(self):
        result = parse_log(self.LOG)
        assert result.error_type == "compiler_error"

    def test_picks_informative_message(self):
        result = parse_log(self.LOG)
        assert "undeclared identifier" in result.error_message

    def test_identifier_extracted(self):
        result = parse_log(self.LOG)
        assert "processData" in result.identifiers


# ---------------------------------------------------------------------------
# 3. Unknown error
# ---------------------------------------------------------------------------

class TestUnknownError:
    """Log with no recognizable error pattern returns error_type='unknown'."""

    LOG = (
        "make[1]: Entering directory '/home/user/project'\n"
        "g++ -c -o main.o main.cpp\n"
        "Something went wrong but no standard pattern here\n"
    )

    def test_error_type_unknown(self):
        result = parse_log(self.LOG)
        assert result.error_type == "unknown"

    def test_identifiers_empty(self):
        result = parse_log(self.LOG)
        assert result.identifiers == []

    def test_still_returns_parsed_log(self):
        result = parse_log(self.LOG)
        assert isinstance(result, ParsedLog)
        assert result.raw_log == self.LOG


# ---------------------------------------------------------------------------
# 4. Multiple error lines — most specific wins
# ---------------------------------------------------------------------------

class TestMultipleErrorLines:
    """Log containing both include and compiler errors.

    include_error is more specific and should be returned.
    """

    LOG = (
        "main.cpp:10:5: error: use of undeclared identifier 'foo'\n"
        "    foo(bar);\n"
        "    ^\n"
        "fatal error: missing_header.h: No such file or directory\n"
        "    #include <missing_header.h>\n"
        "compilation terminated.\n"
    )

    def test_most_specific_error_wins(self):
        result = parse_log(self.LOG)
        # include_error is checked before compiler_error in priority order
        assert result.error_type == "include_error"

    def test_identifiers_from_all_errors(self):
        """Even though error_type is include_error, compiler identifiers
        should still be extracted."""
        result = parse_log(self.LOG)
        assert "foo" in result.identifiers


# ---------------------------------------------------------------------------
# 5. Deeply-namespaced identifiers (e.g. llvm::SelectionDAG::getNode)
# ---------------------------------------------------------------------------

class TestNamespacedIdentifiers:
    """Real LLVM-style log with deeply nested namespace qualifiers.

    The identifier regex must capture the full qualified name including
    all namespace levels (llvm::SelectionDAG::getNode) and must NOT
    include the mangled parameter list.
    """

    LOG = (
        "/usr/bin/ld: error: undefined reference to "
        "`llvm::SelectionDAG::getNode(unsigned int, llvm::SDLoc const&, llvm::EVT)'\n"
        "collect2: error: ld returned 1 exit status\n"
    )

    def test_error_type(self):
        result = parse_log(self.LOG)
        assert result.error_type == "linker_error"

    def test_full_namespace_captured(self):
        result = parse_log(self.LOG)
        assert "llvm::SelectionDAG::getNode" in result.identifiers

    def test_parameter_list_stripped(self):
        """The mangled parameter list must NOT appear in any identifier."""
        result = parse_log(self.LOG)
        for ident in result.identifiers:
            assert "(" not in ident
            assert "unsigned" not in ident
