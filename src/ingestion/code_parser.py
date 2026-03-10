"""
Code Parser — Tree-sitter based C++ function extractor.

Parses a C++ repository into function-level chunks.
Each chunk represents one function/method with its metadata.

See docs/2_system_architecture.md §1 for spec.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx"}


@dataclass
class Chunk:
    """A single function-level code chunk extracted from a C++ file."""
    chunk_id: str       # "{file_path}::{function_name}"
    file_path: str      # relative to repo root
    function_name: str
    start_line: int
    end_line: int
    code_text: str
    language: str = "cpp"


def parse_repository(repo_path: Path) -> List[Chunk]:
    """Parse all C++ files in a repository into function-level chunks.

    Args:
        repo_path: Path to the root of the C++ repository.

    Returns:
        List of Chunk objects, one per function found.
    """
    raise NotImplementedError


def parse_file(file_path: Path, repo_root: Path) -> List[Chunk]:
    """Parse a single C++ file into function-level chunks.

    Args:
        file_path: Absolute path to the C++ file.
        repo_root: Root of the repository (for computing relative paths).

    Returns:
        List of Chunk objects. Empty list if file cannot be parsed.
    """
    raise NotImplementedError
