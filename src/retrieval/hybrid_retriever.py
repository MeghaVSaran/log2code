"""
Hybrid Retriever — fuses dense (ChromaDB) and sparse (BM25) search.

Score fusion: final = 0.5 * dense_score + 0.5 * bm25_score
Optional file-path boost: results matching extracted source paths get +0.5.
Optional file-path dedup: keeps highest-scoring result per unique file.

See docs/2_system_architecture.md §7 for spec.
"""

from dataclasses import dataclass
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5
DEFAULT_CANDIDATES = 20   # fetch this many from each index before fusion
DEFAULT_TOP_K = 5
SOURCE_PATH_BOOST = 0.5    # score bonus for results matching extracted source paths


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
        source_paths: List[str] = None,
        deduplicate_files: bool = True,
    ) -> List[RetrievalResult]:
        """Retrieve top-k most relevant code chunks for a log.

        Queries both the dense (ChromaDB) and sparse (BM25) indices,
        normalises their scores to [0, 1], and fuses them with weights
        ``0.5 * dense + 0.5 * bm25``.

        Falls back to a single index if the other is unavailable.

        Args:
            log_embedding: 768-dim vector from log embedder.
            log_text: Raw query text for BM25 (error_message + identifiers).
            top_k: Number of results to return.
            source_paths: Optional list of normalized source file paths
                extracted from the error log.  Results whose ``file_path``
                matches any of these get a score boost.
            deduplicate_files: If True, collapse results so that each
                unique ``file_path`` appears at most once (keeping the
                highest-scoring function per file).

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

        # --- file-path boost -----------------------------------------------
        if source_paths:
            fused = self._apply_source_path_boost(fused, source_paths)

        # Sort descending by fused score, assign ranks.
        fused.sort(key=lambda r: r.score, reverse=True)

        # --- file-path deduplication ---------------------------------------
        if deduplicate_files:
            fused = self._deduplicate_files(fused)

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

    def _apply_source_path_boost(
        self,
        results: List[RetrievalResult],
        source_paths: List[str],
    ) -> List[RetrievalResult]:
        """Boost scores of results whose file_path matches an extracted source path.

        Matching is done by checking if the result's file_path ends with
        any of the source_paths (or vice versa), or if they share the same
        basename when no relative-path match is possible.

        Args:
            results: Fused results (not yet sorted).
            source_paths: Normalized paths from extract_source_paths().

        Returns:
            Same results list with boosted scores.
        """
        if not source_paths:
            return results

        # Build a set of basenames for fallback matching
        source_basenames = {p.rsplit('/', 1)[-1] for p in source_paths}

        for r in results:
            # Normalize to forward slashes for comparison
            fp = r.file_path.replace('\\', '/')

            matched = False
            for sp in source_paths:
                # Check if one is a suffix of the other
                if fp.endswith(sp) or sp.endswith(fp):
                    matched = True
                    break

            if not matched:
                # Fallback: basename match
                fp_basename = fp.rsplit('/', 1)[-1]
                if fp_basename in source_basenames:
                    matched = True

            if matched:
                r.score = min(r.score + SOURCE_PATH_BOOST, 1.0)

        return results

    def _deduplicate_files(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Keep only the highest-scoring result per unique file_path.

        Assumes results are already sorted by score descending.

        Args:
            results: Sorted list of RetrievalResult.

        Returns:
            Deduplicated list preserving sort order.
        """
        seen: set = set()
        deduped: List[RetrievalResult] = []
        for r in results:
            if r.file_path not in seen:
                seen.add(r.file_path)
                deduped.append(r)
        return deduped
