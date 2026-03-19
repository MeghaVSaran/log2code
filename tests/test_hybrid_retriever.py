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
)
from src.indexing.vector_index import IndexNotFoundError


# ---------------------------------------------------------------------------
# Fake index classes
# ---------------------------------------------------------------------------

class FakeVectorIndex:
    """Returns canned results for query()."""

    def __init__(self, results=None, raise_error=False):
        self._results = results or []
        self._raise_error = raise_error

    def query(self, log_embedding, top_k=20):
        if self._raise_error:
            raise IndexNotFoundError("No collection")
        return self._results


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


class TestSourcePathBoost:
    """Fix 4: Source path boost increases scores for matching file paths."""

    def test_boost_increases_matching_score(self):
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index([]),
        )
        # Without boost
        results_no_boost = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=None, deduplicate_files=False,
        )
        score_a_no_boost = [r for r in results_no_boost if r.file_path == "a.cpp"][0].score

        # With boost matching a.cpp
        results_boosted = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["a.cpp"], deduplicate_files=False,
        )
        score_a_boosted = [r for r in results_boosted if r.file_path == "a.cpp"][0].score

        assert score_a_boosted > score_a_no_boost

    def test_boost_can_change_ranking(self):
        """A low-scoring result can jump to #1 if its path matches."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index([]),
        )
        # c.cpp has the lowest dense score (0.5)
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["c.cpp"], deduplicate_files=False,
        )
        # c.cpp should be boosted to rank 1
        assert results[0].file_path == "c.cpp"

    def test_boost_no_match_unchanged(self):
        """Non-matching results should not be affected."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index([]),
        )
        results_no_boost = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=None, deduplicate_files=False,
        )
        results_boosted = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["nonexistent.cpp"], deduplicate_files=False,
        )
        # Scores should be identical for all results
        for r1, r2 in zip(
            sorted(results_no_boost, key=lambda r: r.chunk_id),
            sorted(results_boosted, key=lambda r: r.chunk_id),
        ):
            assert abs(r1.score - r2.score) < 1e-6

    def test_suffix_matching(self):
        """Boost should work with suffix matching (a.cpp matches dir/a.cpp)."""
        retriever = HybridRetriever(
            FakeVectorIndex(DENSE_RESULTS),
            FakeBM25Index([]),
        )
        results = retriever.retrieve(
            FAKE_EMBEDDING, "test", top_k=10,
            source_paths=["path/to/a.cpp"], deduplicate_files=False,
        )
        a_result = [r for r in results if r.file_path == "a.cpp"][0]
        # a.cpp should be boosted because "a.cpp" is a suffix of "path/to/a.cpp"
        assert a_result.score > 0.5  # Would be 0.5 (dense=1.0 * 0.5) without boost
