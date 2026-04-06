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
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5
DEFAULT_CANDIDATES = 20   # fetch this many from each index before fusion
DEFAULT_TOP_K = 5
SOURCE_PATH_SCORE = 0.95   # fixed score for directly-fetched source path chunks
SYMBOL_CANDIDATES = 20
HINT_CANDIDATES = 20
PREFIX_CANDIDATES = 20


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
    symbol_name: str = ""
    class_name: str = ""
    namespace: str = ""
    signature: str = ""
    symbol_score: float = 0.0


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
        mode: str = "hybrid",
        path_boost: bool = True,
        parsed_log=None,
        strict_pathless: bool = False,
    ) -> List[RetrievalResult]:
        """Retrieve top-k most relevant code chunks for a log.

        Queries both the dense (ChromaDB) and sparse (BM25) indices,
        normalises their scores to [0, 1], and fuses them with weights
        ``0.5 * dense + 0.5 * bm25``.

        When ``source_paths`` are provided and ``path_boost`` is True,
        chunks from those files are fetched directly from ChromaDB via
        metadata filtering and injected into the candidate pool with a
        high fixed score (0.95).

        Falls back to a single index if the other is unavailable.

        Args:
            log_embedding: 768-dim vector from log embedder.
            log_text: Raw query text for BM25 (error_message + identifiers).
            top_k: Number of results to return.
            source_paths: Optional list of normalized source file paths
                extracted from the error log.
            deduplicate_files: If True, collapse results so that each
                unique ``file_path`` appears at most once.
            mode: Retrieval mode — ``"hybrid"`` (default), ``"bm25"``,
                or ``"dense"``.  Controls which indices are queried.
            path_boost: If True (default), inject directly-fetched
                source path chunks.  Set to False to disable.

        Returns:
            List of RetrievalResult sorted by descending fused score.
        """
        # --- dense results -------------------------------------------------
        dense_results: List[Dict] = []
        if mode in ("hybrid", "dense"):
            try:
                dense_results = self.vector_index.query(
                    log_embedding, top_k=DEFAULT_CANDIDATES
                )
            except Exception as exc:
                logger.warning("Dense index failed: %s", exc)

        # --- sparse results ------------------------------------------------
        bm25_results: List[Dict] = []
        if mode in ("hybrid", "bm25"):
            bm25_results = self.bm25_index.query(
                log_text, top_k=DEFAULT_CANDIDATES
            )
            symbol_results = self._get_symbol_candidates(parsed_log)
            if symbol_results:
                bm25_results = self._merge_candidates(bm25_results, symbol_results)
            hint_results = self._get_hint_candidates(parsed_log)
            if hint_results:
                bm25_results = self._merge_candidates(bm25_results, hint_results)

        dense_weight, sparse_weight = self._select_fusion_weights(
            parsed_log,
            strict_pathless=strict_pathless,
        )

        # --- fuse ----------------------------------------------------------
        fused = self._fuse(
            dense_results,
            bm25_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        # --- direct source-path injection ----------------------------------
        if path_boost and source_paths:
            injected = self._fetch_source_path_chunks(
                log_embedding, source_paths
            )
            fused = self._inject_source_chunks(fused, injected)

        if parsed_log is not None:
            fused = self._apply_symbol_boosts(
                fused,
                parsed_log,
                strict_pathless=strict_pathless,
            )

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
        if abs(mx - mn) < 1e-9:
            return [1.0] * len(scores)
        denom = mx - mn
        return [(s - mn) / denom for s in scores]

    def _fuse(
        self,
        dense_results: List[Dict],
        bm25_results: List[Dict],
        dense_weight: float = DENSE_WEIGHT,
        sparse_weight: float = SPARSE_WEIGHT,
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
            fused = dense_weight * d_score + sparse_weight * b_score

            info = meta[cid]
            results.append(RetrievalResult(
                rank=0,  # assigned by caller after sorting
                chunk_id=cid,
                file_path=info["file_path"],
                function_name=info["function_name"],
                symbol_name=info.get("symbol_name", ""),
                class_name=info.get("class_name", ""),
                namespace=info.get("namespace", ""),
                signature=info.get("signature", ""),
                start_line=info["start_line"],
                score=fused,
                dense_score=d_score,
                bm25_score=b_score,
                symbol_score=0.0,
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
                symbol_name=r.get("symbol_name", ""),
                class_name=r.get("class_name", ""),
                namespace=r.get("namespace", ""),
                signature=r.get("signature", ""),
                start_line=r["start_line"],
                score=SOURCE_PATH_SCORE,
                dense_score=SOURCE_PATH_SCORE,
                bm25_score=0.0,
                symbol_score=0.0,
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

    def _get_symbol_candidates(self, parsed_log) -> List[Dict]:
        """Ask BM25 index for exact symbol-based candidates."""
        if parsed_log is None:
            return []
        if not hasattr(self.bm25_index, "query_by_symbols"):
            return []

        identifiers = list(getattr(parsed_log, "identifiers", []))
        if not identifiers:
            return []
        try:
            return self.bm25_index.query_by_symbols(
                identifiers,
                top_k=SYMBOL_CANDIDATES,
            )
        except Exception as exc:
            logger.warning("Symbol candidate query failed: %s", exc)
            return []

    def _merge_candidates(
        self,
        primary: List[Dict],
        extra: List[Dict],
    ) -> List[Dict]:
        """Merge candidate dicts by chunk id, keeping max score."""
        merged: Dict[str, Dict] = {}
        for row in primary + extra:
            cid = row["chunk_id"]
            if cid not in merged or row.get("score", 0.0) > merged[cid].get("score", 0.0):
                merged[cid] = row
        return sorted(
            merged.values(),
            key=lambda r: r.get("score", 0.0),
            reverse=True,
        )[:DEFAULT_CANDIDATES]

    def _get_hint_candidates(self, parsed_log) -> List[Dict]:
        """Retrieve additional candidates from normalized path/file hints."""
        if parsed_log is None:
            return []
        hints = list(getattr(parsed_log, "source_paths", [])) + list(
            getattr(parsed_log, "file_hints", [])
        )
        # Keep only path-like hints with at least one slash.
        hint_terms = [h for h in hints if isinstance(h, str) and ("/" in h or "\\" in h)]
        if not hint_terms:
            return []
        query_text = " ".join(hint_terms)
        if not query_text.strip():
            return []
        try:
            results = self.bm25_index.query(query_text, top_k=HINT_CANDIDATES)
            prefixes = self._derive_path_prefixes(parsed_log)
            if prefixes and hasattr(self.bm25_index, "query_by_path_prefix"):
                prefix_results = self.bm25_index.query_by_path_prefix(
                    prefixes,
                    top_k=PREFIX_CANDIDATES,
                )
                results = self._merge_candidates(results, prefix_results)
            return results
        except Exception as exc:
            logger.warning("Hint candidate query failed: %s", exc)
            return []

    def _select_fusion_weights(
        self,
        parsed_log,
        strict_pathless: bool = False,
    ) -> tuple[float, float]:
        """Choose dense/sparse fusion weights based on query structure."""
        dense_weight = DENSE_WEIGHT
        sparse_weight = SPARSE_WEIGHT
        if parsed_log is None:
            return dense_weight, sparse_weight

        error_type = getattr(parsed_log, "error_type", "unknown")
        has_identifiers = bool(getattr(parsed_log, "identifiers", []))
        has_stack = bool(getattr(parsed_log, "stack_frames", []))
        has_hints = bool(getattr(parsed_log, "source_paths", [])) or bool(
            getattr(parsed_log, "file_hints", [])
        )

        if strict_pathless:
            if error_type == "linker_error":
                sparse_weight = 0.92 if has_identifiers else 0.84
            elif error_type in {"compiler_error", "include_error", "template_error", "build_system_error"}:
                sparse_weight = 0.86 if has_identifiers else 0.78
            elif error_type in {"segfault", "asan_error"}:
                sparse_weight = 0.80 if (has_stack or has_identifiers) else 0.70
            elif error_type == "runtime_exception":
                sparse_weight = 0.72 if has_identifiers else 0.64
            dense_weight = 1.0 - sparse_weight
            return dense_weight, sparse_weight

        if error_type in {"compiler_error", "linker_error", "include_error", "template_error", "build_system_error"}:
            sparse_weight = 0.80 if has_identifiers else 0.72
            dense_weight = 1.0 - sparse_weight
        elif error_type in {"segfault", "asan_error", "runtime_exception"}:
            sparse_weight = 0.65 if (has_stack or has_identifiers) else 0.58
            dense_weight = 1.0 - sparse_weight

        if has_hints:
            sparse_weight = min(0.90, sparse_weight + 0.05)
            dense_weight = 1.0 - sparse_weight

        return dense_weight, sparse_weight

    def _apply_symbol_boosts(
        self,
        results: List[RetrievalResult],
        parsed_log,
        strict_pathless: bool = False,
    ) -> List[RetrievalResult]:
        """Boost results whose symbol metadata matches parsed log evidence."""
        full_identifiers = {
            self._normalize_symbol(identifier)
            for identifier in getattr(parsed_log, "identifiers", [])
            if identifier
        }
        base_identifiers = {
            ident.split("::")[-1]
            for ident in full_identifiers
            if ident
        }
        stack_symbols = {
            self._normalize_symbol(frame)
            for frame in getattr(parsed_log, "stack_frames", [])
            if frame
        }

        if not full_identifiers and not stack_symbols:
            path_prefixes = self._derive_path_prefixes(parsed_log)
            if not path_prefixes:
                return results
        else:
            path_prefixes = self._derive_path_prefixes(parsed_log)

        for result in results:
            function_name = self._normalize_symbol(result.function_name)
            symbol_name = self._normalize_symbol(
                getattr(result, "symbol_name", "") or result.function_name.split("::")[-1]
            )
            signature = self._normalize_symbol(getattr(result, "signature", ""))
            file_stem = self._normalize_symbol(Path(result.file_path).stem)
            boost = 0.0

            error_type = getattr(parsed_log, "error_type", "unknown")
            strict_symbol_mode = strict_pathless and error_type in {"linker_error", "segfault", "asan_error"}

            if strict_symbol_mode:
                if function_name in full_identifiers:
                    boost += 0.58
                if function_name in stack_symbols:
                    boost += 0.35
                if symbol_name in base_identifiers:
                    boost += 0.28
                if signature and any(identifier in signature for identifier in full_identifiers):
                    boost += 0.12
                if file_stem and file_stem in base_identifiers:
                    boost += 0.03
                max_boost = 0.90
            else:
                if function_name in full_identifiers:
                    boost += 0.45
                if function_name in stack_symbols:
                    boost += 0.25
                if symbol_name in base_identifiers:
                    boost += 0.20
                if signature and any(identifier in signature for identifier in full_identifiers):
                    boost += 0.08
                if file_stem and file_stem in base_identifiers:
                    boost += 0.05
                if path_prefixes and any(
                    result.file_path == pref or result.file_path.startswith(f"{pref}/")
                    for pref in path_prefixes
                ):
                    boost += 0.10
                max_boost = 0.75

            result.symbol_score = min(boost, max_boost)
            result.score = min(1.0, result.score + result.symbol_score)

        return results

    def _normalize_symbol(self, value: str) -> str:
        """Lowercase and strip non-symbol prefix noise from symbol strings."""
        value = value or ""
        value = value.strip().lower()
        if " in " in value:
            value = value.split(" in ", 1)[1]
        if "(" in value:
            value = value.split("(", 1)[0]
        return value.strip("`'\" ")

    def _derive_path_prefixes(self, parsed_log) -> List[str]:
        """Derive component prefixes from parsed path hints."""
        raw_paths = list(getattr(parsed_log, "source_paths", [])) + list(
            getattr(parsed_log, "file_hints", [])
        )
        prefixes: set[str] = set()
        for path in raw_paths:
            if not isinstance(path, str):
                continue
            path = path.replace("\\", "/").strip("/")
            if not path or "/" not in path:
                continue
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2:
                prefixes.add("/".join(parts[:2]))
            if len(parts) >= 3:
                prefixes.add("/".join(parts[:3]))
        return sorted(prefixes)
