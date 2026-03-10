"""
Hybrid Retriever — fuses dense (ChromaDB) and sparse (BM25) search.

Score fusion: final = 0.6 * dense_score + 0.4 * bm25_score

See docs/2_system_architecture.md §7 for spec.
"""

from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)

DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4
DEFAULT_CANDIDATES = 20   # fetch this many from each index before fusion
DEFAULT_TOP_K = 5


@dataclass
class RetrievalResult:
    """A single ranked result from hybrid retrieval."""
    rank: int
    chunk_id: str
    file_path: str
    function_name: str
    start_line: int
    score: float           # fused score
    dense_score: float
    bm25_score: float


class HybridRetriever:
    """Retrieves relevant code chunks for a given log using hybrid search."""

    def __init__(self, vector_index, bm25_index):
        """
        Args:
            vector_index: Initialized VectorIndex instance.
            bm25_index: Initialized BM25Index instance.
        """
        self.vector_index = vector_index
        self.bm25_index = bm25_index

    def retrieve(
        self,
        log_embedding,
        log_text: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[RetrievalResult]:
        """Retrieve top-k most relevant code chunks for a log.

        Args:
            log_embedding: 768-dim vector from log embedder.
            log_text: Raw query text for BM25 (error_message + identifiers).
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by descending score.
        """
        raise NotImplementedError

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize a list of scores to [0, 1] range."""
        raise NotImplementedError

    def _fuse(self, dense_results, bm25_results) -> List[RetrievalResult]:
        """Merge and fuse dense and sparse results.

        Args:
            dense_results: Top-N results from vector index.
            bm25_results: Top-N results from BM25 index.

        Returns:
            Combined results with fused scores.
        """
        raise NotImplementedError
