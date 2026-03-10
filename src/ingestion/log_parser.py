"""
Log Parser — Regex-based C++ error log extractor.

Parses raw build/runtime log text into structured ParsedLog objects.
Handles 5 error categories: linker, compiler, include, template, segfault.

See docs/2_system_architecture.md §2 for spec.
"""

from dataclasses import dataclass, field
from typing import List
import re
import logging

logger = logging.getLogger(__name__)

ERROR_TYPES = {
    "linker_error",
    "compiler_error",
    "include_error",
    "template_error",
    "segfault",
    "unknown",
}


@dataclass
class ParsedLog:
    """Structured representation of a parsed C++ error log."""
    raw_log: str
    error_type: str                     # one of ERROR_TYPES
    error_message: str                  # single most informative line
    identifiers: List[str] = field(default_factory=list)   # symbol names
    file_hints: List[str] = field(default_factory=list)    # filenames mentioned
    stack_frames: List[str] = field(default_factory=list)  # segfault frame lines

    def query_text(self) -> str:
        """Return text to use as retrieval query (log embedder input)."""
        parts = [self.error_message] + self.identifiers
        return " ".join(parts)


def parse_log(log_text: str) -> ParsedLog:
    """Parse a raw C++ error log into a structured ParsedLog.

    Always returns a ParsedLog even if parsing is partial.

    Args:
        log_text: Raw log string (may be multi-line).

    Returns:
        ParsedLog with all available fields populated.
    """
    raise NotImplementedError


def extract_error_type(log_text: str) -> str:
    """Classify the primary error type in a log.

    Args:
        log_text: Raw log string.

    Returns:
        One of: linker_error, compiler_error, include_error,
                template_error, segfault, unknown.
    """
    raise NotImplementedError


def extract_identifiers(log_text: str) -> List[str]:
    """Extract C++ symbol names from a log.

    Looks for patterns like:
    - 'undefined reference to Parser::resolveSymbol'
    - 'use of undeclared identifier resolveSymbol'
    - '#0  Parser::resolveSymbol' (stack frames)

    Args:
        log_text: Raw log string.

    Returns:
        List of unique identifier strings found.
    """
    raise NotImplementedError


def extract_file_hints(log_text: str) -> List[str]:
    """Extract filenames mentioned in a log.

    Args:
        log_text: Raw log string.

    Returns:
        List of filename strings (e.g. ['resolve.cpp', 'symbol_table.h']).
    """
    raise NotImplementedError
