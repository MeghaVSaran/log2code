"""
Tests for src/evaluation/metrics.py

Tests recall_at_k, mrr_score, and evaluate_dataset using small
synthetic data and fake retriever/embedder/parser objects.
"""

import pytest
from dataclasses import dataclass
from typing import List

from src.evaluation.metrics import recall_at_k, mrr_score, evaluate_dataset, EvalReport


# ---------------------------------------------------------------------------
# recall_at_k tests
# ---------------------------------------------------------------------------

class TestRecallAtK:
    GROUND_TRUTH = ["target.cpp", "target.h"]

    def test_hit_at_rank_1(self):
        preds = ["target.cpp", "other1.cpp", "other2.cpp"]
        assert recall_at_k(preds, self.GROUND_TRUTH, k=1) == 1.0

    def test_hit_at_rank_5(self):
        preds = ["a.cpp", "b.cpp", "c.cpp", "d.cpp", "target.h"]
        assert recall_at_k(preds, self.GROUND_TRUTH, k=5) == 1.0

    def test_miss_all(self):
        preds = ["a.cpp", "b.cpp", "c.cpp"]
        assert recall_at_k(preds, self.GROUND_TRUTH, k=3) == 0.0

    def test_hit_at_rank_3_but_k_is_2(self):
        preds = ["a.cpp", "b.cpp", "target.cpp"]
        assert recall_at_k(preds, self.GROUND_TRUTH, k=2) == 0.0

    def test_empty_predictions(self):
        assert recall_at_k([], self.GROUND_TRUTH, k=5) == 0.0

    def test_empty_ground_truth(self):
        assert recall_at_k(["a.cpp", "b.cpp"], [], k=5) == 0.0

    def test_both_empty(self):
        assert recall_at_k([], [], k=5) == 0.0

    def test_k_larger_than_predictions(self):
        """k=10 but only 2 predictions — should still work."""
        preds = ["a.cpp", "target.cpp"]
        assert recall_at_k(preds, self.GROUND_TRUTH, k=10) == 1.0


# ---------------------------------------------------------------------------
# mrr_score tests
# ---------------------------------------------------------------------------

class TestMRR:
    GROUND_TRUTH = ["target.cpp", "target.h"]

    def test_first_rank(self):
        preds = ["target.cpp", "a.cpp", "b.cpp"]
        assert mrr_score(preds, self.GROUND_TRUTH) == 1.0

    def test_second_rank(self):
        preds = ["a.cpp", "target.h", "b.cpp"]
        assert mrr_score(preds, self.GROUND_TRUTH) == 0.5

    def test_third_rank(self):
        preds = ["a.cpp", "b.cpp", "target.cpp"]
        assert abs(mrr_score(preds, self.GROUND_TRUTH) - 1.0 / 3) < 1e-9

    def test_no_match(self):
        preds = ["a.cpp", "b.cpp", "c.cpp"]
        assert mrr_score(preds, self.GROUND_TRUTH) == 0.0

    def test_empty_predictions(self):
        assert mrr_score([], self.GROUND_TRUTH) == 0.0

    def test_empty_ground_truth(self):
        assert mrr_score(["a.cpp"], []) == 0.0


# ---------------------------------------------------------------------------
# evaluate_dataset tests
# ---------------------------------------------------------------------------

# Fake objects to avoid real model dependencies.

@dataclass
class FakeParsedLog:
    error_type: str = "linker_error"
    error_message: str = "undefined reference to foo"
    identifiers: list = None

    def __post_init__(self):
        if self.identifiers is None:
            self.identifiers = ["foo"]

    def query_text(self) -> str:
        return self.error_message + " " + " ".join(self.identifiers)


class FakeLogParser:
    def parse_log(self, text):
        return FakeParsedLog()


class FakeLogEmbedder:
    def embed_log(self, parsed_log):
        return [0.0] * 768


@dataclass
class FakeResult:
    rank: int
    chunk_id: str
    file_path: str
    function_name: str
    start_line: int
    score: float
    dense_score: float
    bm25_score: float


class FakeRetriever:
    """Returns canned results for every query."""

    def __init__(self, file_paths: List[str]):
        self._paths = file_paths

    def retrieve(self, log_embedding, log_text, top_k=5):
        return [
            FakeResult(
                rank=i + 1,
                chunk_id=f"{fp}::func",
                file_path=fp,
                function_name="func",
                start_line=1,
                score=1.0 / (i + 1),
                dense_score=1.0 / (i + 1),
                bm25_score=0.0,
            )
            for i, fp in enumerate(self._paths[:top_k])
        ]


class TestEvaluateDataset:

    def test_perfect_retrieval(self):
        """Retriever always returns the target file first."""
        dataset = [
            {"log": "error", "relevant_files": ["a.cpp"], "error_type": "linker_error"},
            {"log": "error", "relevant_files": ["a.cpp"], "error_type": "linker_error"},
        ]
        retriever = FakeRetriever(["a.cpp", "b.cpp", "c.cpp"])
        report = evaluate_dataset(dataset, retriever, FakeLogParser(), FakeLogEmbedder())

        assert isinstance(report, EvalReport)
        assert report.recall_at_1 == 1.0
        assert report.recall_at_5 == 1.0
        assert report.mrr == 1.0
        assert report.num_samples == 2

    def test_miss(self):
        """Retriever never returns the target file."""
        dataset = [
            {"log": "error", "relevant_files": ["z.cpp"], "error_type": "compiler_error"},
        ]
        retriever = FakeRetriever(["a.cpp", "b.cpp", "c.cpp"])
        report = evaluate_dataset(dataset, retriever, FakeLogParser(), FakeLogEmbedder())

        assert report.recall_at_1 == 0.0
        assert report.recall_at_5 == 0.0
        assert report.mrr == 0.0

    def test_per_error_type_breakdown(self):
        dataset = [
            {"log": "err1", "relevant_files": ["a.cpp"], "error_type": "linker_error"},
            {"log": "err2", "relevant_files": ["z.cpp"], "error_type": "compiler_error"},
        ]
        retriever = FakeRetriever(["a.cpp", "b.cpp"])
        report = evaluate_dataset(dataset, retriever, FakeLogParser(), FakeLogEmbedder())

        assert "linker_error" in report.per_error_type
        assert "compiler_error" in report.per_error_type
        assert report.per_error_type["linker_error"]["recall_at_1"] == 1.0
        assert report.per_error_type["compiler_error"]["recall_at_1"] == 0.0

    def test_empty_dataset(self):
        report = evaluate_dataset([], FakeRetriever([]), FakeLogParser(), FakeLogEmbedder())
        assert report.num_samples == 0
        assert report.recall_at_1 == 0.0
