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

# ---------------------------------------------------------------------------
# Compiled regex patterns — grouped by error category.
# Order within _ERROR_PATTERNS matters: first match wins for error_type.
# We check most-specific categories first so that e.g. an include error
# is not accidentally classified as a generic compiler error.
# ---------------------------------------------------------------------------

# Identifier character class: captures qualified C++ names like
# llvm::SelectionDAG::getNode  but stops *before* an opening parenthesis
# so that mangled parameter lists are stripped automatically.
_IDENT = r"[A-Za-z_~][A-Za-z0-9_:<>~*]*"

# --- include error ---
_RE_INCLUDE_ERROR = re.compile(
    r"fatal error:\s*(\S+)\s*:\s*No such file or directory",
    re.IGNORECASE,
)

# --- template error ---
_RE_TEMPLATE_ERROR = re.compile(
    r"(?:implicit instantiation of undefined template|undefined template)",
    re.IGNORECASE,
)

# --- linker error ---
_RE_UNDEF_REF = re.compile(
    r"undefined reference to\s+[`'\"]?(" + _IDENT + r")[`'\"]?",
)
_RE_MULTI_DEF = re.compile(
    r"multiple definition of\s+[`'\"]?(" + _IDENT + r")[`'\"]?",
)

# --- compiler error ---
_RE_UNDECLARED_IDENT = re.compile(
    r"use of undeclared identifier\s+[`'\"]?(" + _IDENT + r")[`'\"]?",
)
_RE_NO_MATCHING_FN = re.compile(
    r"no matching function for call to\s+[`'\"]?(" + _IDENT + r")[`'\"]?",
)

# --- segfault / stack trace ---
_RE_SEGFAULT = re.compile(r"Segmentation fault", re.IGNORECASE)
_RE_STACK_FRAME = re.compile(
    r"#(\d+)\s+(?:0x[0-9a-fA-F]+\s+in\s+)?(" + _IDENT + r")",
)

# --- file hints ---
_RE_FILE_HINT = re.compile(
    r"([\w.\/\\-]+\.(?:cpp|h|hpp|cc|cxx|c|hxx))\b",
)

# Ordered list used by extract_error_type — first match wins.
_ERROR_PATTERNS = [
    ("include_error", _RE_INCLUDE_ERROR),
    ("template_error", _RE_TEMPLATE_ERROR),
    ("linker_error", _RE_UNDEF_REF),
    ("linker_error", _RE_MULTI_DEF),
    ("compiler_error", _RE_UNDECLARED_IDENT),
    ("compiler_error", _RE_NO_MATCHING_FN),
    ("segfault", _RE_SEGFAULT),
    ("segfault", _RE_STACK_FRAME),
]


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
        """Return text to use as retrieval query (log embedder input).

        Format: error_message + space-joined identifiers + file hints.
        """
        parts = [self.error_message] + self.identifiers + self.file_hints
        return " ".join(parts)


# ---- public API -----------------------------------------------------------

def parse_log(log_text: str) -> ParsedLog:
    """Parse a raw C++ error log into a structured ParsedLog.

    Always returns a ParsedLog even if parsing is partial.

    Args:
        log_text: Raw log string (may be multi-line).

    Returns:
        ParsedLog with all available fields populated.
    """
    error_type = extract_error_type(log_text)
    identifiers = extract_identifiers(log_text)
    file_hints = extract_file_hints(log_text)
    error_message = _pick_error_message(log_text, error_type)
    stack_frames = _extract_stack_frames(log_text)

    parsed = ParsedLog(
        raw_log=log_text,
        error_type=error_type,
        error_message=error_message,
        identifiers=identifiers,
        file_hints=file_hints,
        stack_frames=stack_frames,
    )
    logger.debug("Parsed log → type=%s, identifiers=%s", error_type, identifiers)
    return parsed


def extract_error_type(log_text: str) -> str:
    """Classify the primary error type in a log.

    Checks patterns in priority order (most-specific first) and returns
    the category of the first match.

    Args:
        log_text: Raw log string.

    Returns:
        One of: linker_error, compiler_error, include_error,
                template_error, segfault, unknown.
    """
    for error_type, pattern in _ERROR_PATTERNS:
        if pattern.search(log_text):
            return error_type
    return "unknown"


def extract_identifiers(log_text: str) -> List[str]:
    """Extract C++ symbol names from a log.

    Looks for patterns like:
    - 'undefined reference to `Parser::resolveSymbol`'
    - 'use of undeclared identifier resolveSymbol'
    - 'multiple definition of SymbolTable::insert'
    - 'no matching function for call to Parser::parse'
    - '#0  Parser::resolveSymbol' (stack frames)

    The regex capture groups intentionally stop before '(' so that
    C++ mangled parameter lists are stripped automatically.

    Args:
        log_text: Raw log string.

    Returns:
        List of unique identifier strings found (order-preserved).
    """
    seen: set = set()
    result: List[str] = []

    # All identifier-bearing patterns with their compiled regexes
    ident_patterns = [
        _RE_UNDEF_REF,
        _RE_MULTI_DEF,
        _RE_UNDECLARED_IDENT,
        _RE_NO_MATCHING_FN,
        _RE_STACK_FRAME,
    ]

    for pattern in ident_patterns:
        for match in pattern.finditer(log_text):
            # For stack frames the identifier is group(2), else group(1)
            if pattern is _RE_STACK_FRAME:
                ident = match.group(2)
            else:
                ident = match.group(1)

            # Strip surrounding quotes / backticks that may have leaked
            ident = ident.strip("`'\"")

            if ident and ident not in seen:
                seen.add(ident)
                result.append(ident)

    return result


def extract_file_hints(log_text: str) -> List[str]:
    """Extract filenames mentioned in a log.

    Catches both general C/C++ filenames and the specific include-error
    pattern (``fatal error: X.h: No such file``).

    Args:
        log_text: Raw log string.

    Returns:
        List of unique filename strings (e.g. ['resolve.cpp', 'symbol_table.h']).
    """
    seen: set = set()
    result: List[str] = []

    # Include-error specific extraction first (higher quality)
    for m in _RE_INCLUDE_ERROR.finditer(log_text):
        fname = m.group(1)
        if fname not in seen:
            seen.add(fname)
            result.append(fname)

    # General C/C++ filename extraction
    for m in _RE_FILE_HINT.finditer(log_text):
        fname = m.group(1)
        if fname not in seen:
            seen.add(fname)
            result.append(fname)

    return result


# ---- internal helpers ------------------------------------------------------

def _pick_error_message(log_text: str, error_type: str) -> str:
    """Select the single most informative line from the log.

    Scans all lines and returns the first one that matches a pattern
    associated with the detected *error_type*.  Falls back to the first
    non-empty line.

    Args:
        log_text: Raw log text.
        error_type: Already-detected error category.

    Returns:
        A single-line string (stripped).
    """
    # Map error_type to its relevant patterns
    type_to_patterns = {
        "include_error": [_RE_INCLUDE_ERROR],
        "template_error": [_RE_TEMPLATE_ERROR],
        "linker_error": [_RE_UNDEF_REF, _RE_MULTI_DEF],
        "compiler_error": [_RE_UNDECLARED_IDENT, _RE_NO_MATCHING_FN],
        "segfault": [_RE_SEGFAULT],
    }

    patterns = type_to_patterns.get(error_type, [])
    first_nonempty = ""

    for line in log_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not first_nonempty:
            first_nonempty = stripped

        for pat in patterns:
            if pat.search(stripped):
                return stripped

    return first_nonempty


def _extract_stack_frames(log_text: str) -> List[str]:
    """Extract stack frame lines from the log (e.g. from segfault traces).

    Args:
        log_text: Raw log text.

    Returns:
        List of raw stack-frame lines, in order.
    """
    frames: List[str] = []
    for line in log_text.splitlines():
        stripped = line.strip()
        if _RE_STACK_FRAME.search(stripped):
            frames.append(stripped)
    return frames
