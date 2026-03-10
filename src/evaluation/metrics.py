"""
Evaluation metrics: Recall@K and MRR.

See docs/3_mvp_spec.md §F6 for spec.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class EvalReport:
    """Aggregate evaluation results across a dataset."""
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    num_samples: int
    per_error_type: Dict[str, Dict[str, float]]  # {error_type: {metric: value}}


def recall_at_k(
    predictions: List[str],
    ground_truth: List[str],
    k: int,
) -> float:
    """Compute Recall@K for a single query.

    A hit is counted if any ground truth file appears in the top-k predictions.

    Args:
        predictions: Ordered list of predicted file paths (best first).
        ground_truth: List of relevant file paths.
        k: Cutoff rank.

    Returns:
        1.0 if any ground truth file is in predictions[:k], else 0.0.
    """
    raise NotImplementedError


def mrr_score(
    predictions: List[str],
    ground_truth: List[str],
) -> float:
    """Compute MRR for a single query.

    Args:
        predictions: Ordered list of predicted file paths (best first).
        ground_truth: List of relevant file paths.

    Returns:
        1/rank of the first correct prediction, or 0.0 if none found.
    """
    raise NotImplementedError


def evaluate_dataset(dataset: List[Dict], retriever) -> EvalReport:
    """Run full evaluation on a ground truth dataset.

    Args:
        dataset: List of dicts with keys: log, relevant_files, error_type.
        retriever: HybridRetriever instance with pre-built indices.

    Returns:
        EvalReport with aggregate and per-category metrics.
    """
    raise NotImplementedError
