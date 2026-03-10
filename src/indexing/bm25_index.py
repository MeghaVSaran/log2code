"""
BM25 Index — rank_bm25 wrapper for sparse retrieval.

See docs/2_system_architecture.md §6 for spec.
"""

from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class BM25Index:
    """Sparse BM25 retrieval over code chunk texts."""

    def __init__(self):
        self._index = None
        self._chunks = []   # parallel list to BM25 corpus

    def build(self, chunks: List) -> None:
        """Build BM25 index from a list of Chunk objects.

        Tokenizes code_text by whitespace + C++ operator stripping.
        Preserves identifier names (CamelCase, snake_case) intact.

        Args:
            chunks: List of Chunk dataclass objects.
        """
        raise NotImplementedError

    def query(self, text: str, top_k: int = 20) -> List[Dict]:
        """Score all chunks against query text.

        Args:
            text: Query string (log error_message + identifiers).
            top_k: Number of results to return.

        Returns:
            List of dicts: {chunk_id, file_path, function_name, start_line, score}
            Score is raw BM25 score (not normalized here).
        """
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Persist index to disk using pickle."""
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """Load index from disk."""
        raise NotImplementedError

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize code text for BM25.

        Splits on whitespace and common C++ operators.
        Keeps identifier names intact.
        """
        raise NotImplementedError
