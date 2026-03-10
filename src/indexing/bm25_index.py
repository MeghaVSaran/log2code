"""
BM25 Index — rank_bm25 wrapper for sparse retrieval.

See docs/2_system_architecture.md §6 for spec.
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging
import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Regex that splits on whitespace and common C++ punctuation / operators.
_SPLIT_RE = re.compile(r"[\s()\{\};,<>*&:.]+|->|::")


class BM25Index:
    """Sparse BM25 retrieval over code chunk texts."""

    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._chunks: List = []   # parallel list to BM25 corpus

    def build(self, chunks: List) -> None:
        """Build BM25 index from a list of Chunk objects.

        Tokenizes code_text by whitespace + C++ operator stripping.
        Preserves identifier names (CamelCase, snake_case) intact.

        Args:
            chunks: List of Chunk dataclass objects.
        """
        self._chunks = list(chunks)
        corpus = [self._tokenize(c.code_text) for c in self._chunks]
        self._index = BM25Okapi(corpus)
        logger.info("Built BM25 index with %d chunks.", len(self._chunks))

    def query(self, text: str, top_k: int = 20) -> List[Dict]:
        """Score all chunks against query text.

        Args:
            text: Query string (log error_message + identifiers).
            top_k: Number of results to return.

        Returns:
            List of dicts: {chunk_id, file_path, function_name,
            start_line, score}.
            Score is raw BM25 score (not normalised).
            Returns empty list if all scores are zero.
        """
        if self._index is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(text)
        if not tokenized_query:
            return []

        scores = self._index.get_scores(tokenized_query)

        # If every score is zero there are no useful results.
        if np.max(scores) == 0.0:
            return []

        # Argsort descending, take top_k.
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break  # remaining are zero or negative
            chunk = self._chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "function_name": chunk.function_name,
                "start_line": chunk.start_line,
                "score": score,
            })

        return results

    def save(self, path: Path) -> None:
        """Persist index to disk using pickle.

        Args:
            path: File path for the pickle output.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"index": self._index, "chunks": self._chunks}, f)
        logger.info("Saved BM25 index to %s.", path)

    def load(self, path: Path) -> None:
        """Load index from disk.

        Args:
            path: File path to the pickled index.
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._index = data["index"]
        self._chunks = data["chunks"]
        logger.info(
            "Loaded BM25 index from %s (%d chunks).", path, len(self._chunks)
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize code text for BM25.

        Splits on whitespace and common C++ operators/punctuation.
        Filters out tokens shorter than 2 characters and pure-numeric
        tokens.  Lowercases everything.  Keeps CamelCase and snake_case
        identifiers intact (just lowered).

        Args:
            text: Raw code or query text.

        Returns:
            List of lowercase token strings.
        """
        raw_tokens = _SPLIT_RE.split(text)
        tokens: List[str] = []
        for tok in raw_tokens:
            tok = tok.strip().lower()
            if len(tok) < 2:
                continue
            if tok.isdigit():
                continue
            tokens.append(tok)
        return tokens
