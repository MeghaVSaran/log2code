"""
Shared pytest fixtures for the log2code test suite.

The session-scoped ``cpp_parser`` fixture initialises the tree-sitter
C++ parser exactly once for the entire test run, avoiding the ~12s
startup cost per ``parse_file`` call.
"""

import pytest
import tree_sitter
import tree_sitter_cpp


@pytest.fixture(scope="session")
def cpp_parser() -> tree_sitter.Parser:
    """Return a tree-sitter Parser for C++ initialised once per session."""
    language = tree_sitter.Language(tree_sitter_cpp.language())
    return tree_sitter.Parser(language)
