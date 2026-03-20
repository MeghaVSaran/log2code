"""
Hybrid Retriever — fuses dense (ChromaDB) and sparse (BM25) search.

Score fusion: final = 0.5 * dense_score + 0.5 * bm25_score
Direct file-path injection: when source paths are extracted from the log,
chunks from those files are fetched directly via ChromaDB metadata filter
and injected into the candidate pool with a high fixed score.
File-path dedup: keeps highest-scoring result per unique file.

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
SOURCE_PATH_SCORE = 0.95   # fixed score for directly-fetched source path chunks


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

        When ``source_paths`` are provided, chunks from those files are
        fetched directly from ChromaDB via metadata filtering and
        injected into the candidate pool with a high fixed score
        (0.95).  This guarantees that when the error log mentions a
        file path, that file always appears in the results.

        Falls back to a single index if the other is unavailable.

        Args:
            log_embedding: 768-dim vector from log embedder.
            log_text: Raw query text for BM25 (error_message + identifiers).
            top_k: Number of results to return.
            source_paths: Optional list of normalized source file paths
                extracted from the error log.  Chunks from these files
                are fetched directly and injected into the candidate pool.
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

        # --- direct source-path injection ----------------------------------
        if source_paths:
            injected = self._fetch_source_path_chunks(
                log_embedding, source_paths
            )
            fused = self._inject_source_chunks(fused, injected)

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

    def _fetch_source_path_chunks(
        self,
        log_embedding,
        source_paths: List[str],
    ) -> List[RetrievalResult]:
        """Fetch chunks directly from ChromaDB for specific file paths.

        Uses ChromaDB's metadata ``where`` filter to find chunks whose
        ``file_path`` matches one of the extracted source paths.  These
        are given a high fixed score so they rank above hybrid results.

        Args:
            log_embedding: 768-dim query vector (used by ChromaDB but
                the ``where`` filter controls which chunks are returned).
            source_paths: Normalized file paths extracted from the log.

        Returns:
            List of RetrievalResult with score = SOURCE_PATH_SCORE.
        """
        try:
            if len(source_paths) == 1:
                where_filter = {"file_path": {"$eq": source_paths[0]}}
            else:
                where_filter = {"file_path": {"$in": source_paths}}

            raw = self.vector_index.query(
                log_embedding,
                top_k=10,
                where=where_filter,
            )
        except Exception as exc:
            logger.warning(
                "Source path chunk fetch failed: %s", exc
            )
            return []

        results: List[RetrievalResult] = []
        for r in raw:
            results.append(RetrievalResult(
                rank=0,
                chunk_id=r["chunk_id"],
                file_path=r["file_path"],
                function_name=r["function_name"],
                start_line=r["start_line"],
                score=SOURCE_PATH_SCORE,
                dense_score=SOURCE_PATH_SCORE,
                bm25_score=0.0,
            ))
        return results

    def _inject_source_chunks(
        self,
        fused: List[RetrievalResult],
        injected: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Inject directly-fetched source path chunks into the fused pool.

        If a chunk already exists in the fused pool, its score is updated
        to the maximum of the existing score and the injected score.
        Otherwise, the chunk is added to the pool.

        Args:
            fused: Existing fused results from hybrid search.
            injected: Chunks from _fetch_source_path_chunks().

        Returns:
            Combined result list.
        """
        existing_ids = {r.chunk_id: r for r in fused}

        for inj in injected:
            if inj.chunk_id in existing_ids:
                # Upgrade score if injection score is higher
                existing = existing_ids[inj.chunk_id]
                if inj.score > existing.score:
                    existing.score = inj.score
            else:
                fused.append(inj)

        return fused

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
