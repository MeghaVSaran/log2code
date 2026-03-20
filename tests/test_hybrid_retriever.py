"""
Tests for src/retrieval/hybrid_retriever.py

Uses simple fake index classes (no real ChromaDB or BM25 needed)
to verify fusion math, fallback behaviour, and sort order.
"""

import pytest
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    RetrievalResult,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    SOURCE_PATH_SCORE,
)
from src.indexing.vector_index import IndexNotFoundError


# ---------------------------------------------------------------------------
# Fake index classes
# ---------------------------------------------------------------------------

class FakeVectorIndex:
    """Returns canned results for query().

    Supports optional ``where`` filter: when ``where`` is provided and
    ``filtered_results`` was set at init, those are returned instead.
    """

    def __init__(self, results=None, raise_error=False, filtered_results=None):
        self._results = results or []
        self._raise_error = raise_error
        self._filtered_results = filtered_results or []

    def query(self, log_embedding, top_k=20, where=None):
        if self._raise_error:
            raise IndexNotFoundError("No collection")
        if where is not None:
            return self._filtered_results[:top_k]
        return self._results[:top_k]


class FakeBM25Index:
    """Returns canned results for query()."""

    def __init__(self, results=None):
        self._results = results or []

    def query(self, text, top_k=20):
        return self._results


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

DENSE_RESULTS = [
    {"chunk_id": "a.cpp::foo", "file_path": "a.cpp",
     "function_name": "foo", "start_line": 10, "score": 0.9},
    {"chunk_id": "b.cpp::bar", "file_path": "b.cpp",
     "function_name": "bar", "start_line": 20, "score": 0.7},
    {"chunk_id": "c.cpp::baz", "file_path": "c.cpp",
     "function_name": "baz", "start_line": 30, "score": 0.5},
]

BM25_RESULTS = [
    {"chunk_id": "b.cpp::bar", "file_path": "b.cpp",
     "function_name": "bar", "start_line": 20, "score": 8.0},
    {"chunk_id": "d.cpp::qux", "file_path": "d.cpp",
     "function_name": "qux", "start_line": 40, "score": 5.0},
    {"chunk_id": "a.cpp::foo", "file_path": "a.cpp",
     "function_name": "foo", "start_line": 10, "score": 2.0},
]

FAKE_EMBEDDING = [0.0] * 768


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFusionMath:
    """Verify that score normalization and fusion weights are correct."""

    def test_scores_in_unit_range(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        for r in results:
            assert 0.0 <= r.score <= 1.0 + 1e-6

    def test_dense_and_bm25_scores_normalized(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        for r in results:
            assert 0.0 <= r.dense_score <= 1.0 + 1e-6
            assert 0.0 <= r.bm25_score <= 1.0 + 1e-6

    def test_fusion_weights(self):
        """Fused score should equal 0.5 * dense + 0.5 * bm25."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        for r in results:
            expected = DENSE_WEIGHT * r.dense_score + SPARSE_WEIGHT * r.bm25_score
            assert abs(r.score - expected) < 1e-6

    def test_overlapping_chunk_fused(self):
        """Chunk b.cpp::bar appears in both; should have non-zero dense and bm25."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        bar = [r for r in results if r.chunk_id == "b.cpp::bar"][0]
        assert bar.dense_score > 0.0
        assert bar.bm25_score > 0.0

    def test_unique_chunk_has_zero_for_missing_index(self):
        """d.cpp::qux only in BM25 → dense_score should be 0."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        qux = [r for r in results if r.chunk_id == "d.cpp::qux"][0]
        assert qux.dense_score == 0.0
        assert qux.bm25_score > 0.0


class TestSorting:
    """Results should be sorted by fused score descending with correct ranks."""

    def test_descending_order(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_top_k_limits(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=2)
        assert len(results) == 2


class TestFallback:
    """Graceful degradation when one index is unavailable."""

    def test_dense_fails_falls_back_to_bm25(self):
        retriever = HybridRetriever(
            FakeVectorIndex(raise_error=True),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=5)
        assert len(results) > 0
        # All dense_scores should be 0 (only BM25 contributed).
        for r in results:
            assert r.dense_score == 0.0
        # At least the top result should have a non-zero bm25_score;
        # the lowest-scoring item normalizes to 0.0 (min-max).
        assert results[0].bm25_score > 0.0

    def test_bm25_empty_falls_back_to_dense(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=5)
        assert len(results) > 0
        for r in results:
            assert r.bm25_score == 0.0
        # At least the top result should have a non-zero dense_score.
        assert results[0].dense_score > 0.0

    def test_both_empty_returns_empty(self):
        retriever = HybridRetriever(
            FakeVectorIndex([]),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=5)
        assert results == []


class TestRetrievalResult:
    """Verify the returned objects are proper RetrievalResult instances."""

    def test_returns_retrieval_results(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=3)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_total_unique_chunks(self):
        """Union of 3 dense + 3 BM25 results (2 overlap) = 4 unique."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index(BM25_RESULTS),
        )
        # With dedup on, each file only appears once — all 4 have unique file_paths
        results = retriever.retrieve(FAKE_EMBEDDING, "bar", top_k=10, deduplicate_files=False)
        assert len(results) == 4


class TestFileDeduplication:
    """Fix 5: File-path deduplication keeps only the best result per file."""

    # Two functions from the same file
    DENSE_SAME_FILE = [
        {"chunk_id": "a.cpp::foo", "file_path": "a.cpp",
         "function_name": "foo", "start_line": 10, "score": 0.9},
        {"chunk_id": "a.cpp::bar", "file_path": "a.cpp",
         "function_name": "bar", "start_line": 50, "score": 0.8},
        {"chunk_id": "b.cpp::baz", "file_path": "b.cpp",
         "function_name": "baz", "start_line": 30, "score": 0.5},
    ]

    def test_dedup_removes_same_file_duplicates(self):
        retriever = HybridRetriever(
            FakeVectorIndex(self.DENSE_SAME_FILE),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "test", top_k=10, deduplicate_files=True)
        file_paths = [r.file_path for r in results]
        assert len(file_paths) == len(set(file_paths)), "Duplicate file_paths in results!"
        assert len(results) == 2  # a.cpp and b.cpp

    def test_dedup_keeps_highest_score(self):
        retriever = HybridRetriever(
            FakeVectorIndex(self.DENSE_SAME_FILE),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "test", top_k=10, deduplicate_files=True)
        a_result = [r for r in results if r.file_path == "a.cpp"][0]
        assert a_result.function_name == "foo"  # foo has higher score (0.9 > 0.8)

    def test_dedup_off_keeps_all(self):
        retriever = HybridRetriever(
            FakeVectorIndex(self.DENSE_SAME_FILE),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(FAKE_EMBEDDING, "test", top_k=10, deduplicate_files=False)
        assert len(results) == 3  # All three results kept


class TestSourcePathInjection:
    """Fix 4: Direct source path chunk injection from ChromaDB."""

    # Chunks that would be returned by the where-filter query
    INJECTED_CHUNKS = [
        {"chunk_id": "target.cc::func_x", "file_path": "target.cc",
         "function_name": "func_x", "start_line": 23, "score": 0.3},
    ]

    def test_injection_adds_missing_file(self):
        """A file not in hybrid results should appear after injection."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=self.INJECTED_CHUNKS),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["target.cc"], deduplicate_files=False,
        )
        file_paths = [r.file_path for r in results]
        assert "target.cc" in file_paths

    def test_injected_chunk_has_high_score(self):
        """Injected chunks should get SOURCE_PATH_SCORE (0.95)."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=self.INJECTED_CHUNKS),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["target.cc"], deduplicate_files=False,
        )
        target = [r for r in results if r.file_path == "target.cc"][0]
        assert target.score == SOURCE_PATH_SCORE

    def test_injected_chunk_ranks_first(self):
        """Injected file should be rank 1 (highest score)."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=self.INJECTED_CHUNKS),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["target.cc"], deduplicate_files=True,
        )
        assert results[0].file_path == "target.cc"
        assert results[0].rank == 1

    def test_no_source_paths_no_injection(self):
        """Without source_paths, no injection occurs."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=self.INJECTED_CHUNKS),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=None, deduplicate_files=False,
        )
        file_paths = [r.file_path for r in results]
        assert "target.cc" not in file_paths

    def test_injection_upgrades_existing_chunk(self):
        """If injected chunk already in fused pool, its score is upgraded."""
        # a.cpp is already in DENSE_RESULTS
        existing_chunk = [
            {"chunk_id": "a.cpp::foo", "file_path": "a.cpp",
             "function_name": "foo", "start_line": 10, "score": 0.2},
        ]
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS, filtered_results=existing_chunk),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["a.cpp"], deduplicate_files=False,
        )
        a_result = [r for r in results if r.file_path == "a.cpp"][0]
        # Score should be upgraded to SOURCE_PATH_SCORE
        assert a_result.score == SOURCE_PATH_SCORE
