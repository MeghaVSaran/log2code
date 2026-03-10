"""
Hybrid Retriever — fuses dense (ChromaDB) and sparse (BM25) search.

Score fusion: final = 0.6 * dense_score + 0.4 * bm25_score

See docs/2_system_architecture.md §7 for spec.
"""

from dataclasses import dataclass
from typing import List, Dict
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
        """Initialise the retriever.

        Args:
            vector_index: Initialised VectorIndex instance.
            bm25_index: Initialised BM25Index instance.
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

        Queries both the dense (ChromaDB) and sparse (BM25) indices,
        normalises their scores to [0, 1], and fuses them with weights
        ``0.6 * dense + 0.4 * bm25``.

        Falls back to a single index if the other is unavailable.

        Args:
            log_embedding: 768-dim vector from log embedder.
            log_text: Raw query text for BM25 (error_message + identifiers).
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by descending fused score.
        """
        # --- dense results -------------------------------------------------
        dense_results: List[Dict] = []
        try:
            dense_results = self.vector_index.query(
                log_embedding, top_k=DEFAULT_CANDIDATES
            )
        except Exception as exc:
            # IndexNotFoundError or any other failure → fall back to BM25.
            logger.warning("Dense index failed, falling back to BM25: %s", exc)

        # --- sparse results ------------------------------------------------
        bm25_results: List[Dict] = self.bm25_index.query(
            log_text, top_k=DEFAULT_CANDIDATES
        )

        # --- fuse ----------------------------------------------------------
        fused = self._fuse(dense_results, bm25_results)

        # Sort descending by fused score, assign ranks, return top_k.
        fused.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(fused):
            r.rank = i + 1

        return fused[:top_k]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize a list of scores to [0, 1] range.

        Uses min-max normalisation:
        ``(score - min) / (max - min + 1e-9)``

        Returns all zeros if the input list is empty.
        """
        if not scores:
            return []
        mn = min(scores)
        mx = max(scores)
        denom = mx - mn + 1e-9
        return [(s - mn) / denom for s in scores]

    def _fuse(
        self, dense_results: List[Dict], bm25_results: List[Dict]
    ) -> List[RetrievalResult]:
        """Merge and fuse dense and sparse results.

        Args:
            dense_results: Top-N dicts from vector index.
            bm25_results: Top-N dicts from BM25 index.

        Returns:
            Combined results with fused scores (unranked — caller sorts).
        """
        # Collect metadata keyed by chunk_id.
        meta: Dict[str, Dict] = {}
        for r in dense_results:
            meta[r["chunk_id"]] = r
        for r in bm25_results:
            if r["chunk_id"] not in meta:
                meta[r["chunk_id"]] = r

        # Build normalised score maps.
        dense_norm = self._normalize_scores(
            [r["score"] for r in dense_results]
        )
        bm25_norm = self._normalize_scores(
            [r["score"] for r in bm25_results]
        )

        dense_map: Dict[str, float] = {
            r["chunk_id"]: n
            for r, n in zip(dense_results, dense_norm)
        }
        bm25_map: Dict[str, float] = {
            r["chunk_id"]: n
            for r, n in zip(bm25_results, bm25_norm)
        }

        # Fuse scores for every unique chunk_id.
        all_ids = set(dense_map.keys()) | set(bm25_map.keys())
        results: List[RetrievalResult] = []

        for cid in all_ids:
            d_score = dense_map.get(cid, 0.0)
            b_score = bm25_map.get(cid, 0.0)
            fused = DENSE_WEIGHT * d_score + SPARSE_WEIGHT * b_score

            info = meta[cid]
            results.append(RetrievalResult(
                rank=0,  # assigned by caller after sorting
                chunk_id=cid,
                file_path=info["file_path"],
                function_name=info["function_name"],
                start_line=info["start_line"],
                score=fused,
                dense_score=d_score,
                bm25_score=b_score,
            ))

        return results
