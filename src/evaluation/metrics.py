"""
Evaluation metrics: Recall@K and MRR.

See docs/3_mvp_spec.md §F6 for spec.
"""

from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Aggregate evaluation results across a dataset."""
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    num_samples: int
    per_error_type: Dict[str, Dict[str, float]] = field(default_factory=dict)


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
    if not predictions or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    top_k = predictions[:k]
    for pred in top_k:
        if pred in gt_set:
            return 1.0
    return 0.0


def mrr_score(
    predictions: List[str],
    ground_truth: List[str],
) -> float:
    """Compute MRR (Mean Reciprocal Rank) for a single query.

    Finds the rank (1-indexed) of the first prediction that appears
    in the ground truth set.

    Args:
        predictions: Ordered list of predicted file paths (best first).
        ground_truth: List of relevant file paths.

    Returns:
        1/rank of the first correct prediction, or 0.0 if none found.
    """
    if not predictions or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    for rank, pred in enumerate(predictions, start=1):
        if pred in gt_set:
            return 1.0 / rank
    return 0.0


def evaluate_dataset(
    dataset: List[Dict],
    retriever,
    log_parser,
    log_embedder,
) -> EvalReport:
    """Run full evaluation on a ground truth dataset.

    For each item, parses the log, embeds it, retrieves top-5, and
    computes Recall@1, Recall@3, Recall@5, and MRR.  Aggregates
    results as means across all items and per error type.

    Args:
        dataset: List of dicts with keys: ``log``, ``relevant_files``,
                 and optionally ``error_type``.
        retriever: HybridRetriever instance with pre-built indices.
        log_parser: Module or object with ``parse_log(text)`` function.
        log_embedder: LogEmbedder instance with ``embed_log(parsed)``
                      method.

    Returns:
        EvalReport with aggregate and per-category metrics.
    """
    # Accumulators.
    r1_scores: List[float] = []
    r3_scores: List[float] = []
    r5_scores: List[float] = []
    mrr_scores: List[float] = []

    # Per error-type accumulators.
    by_type: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"recall_at_1": [], "recall_at_3": [],
                 "recall_at_5": [], "mrr": []}
    )

    for item in dataset:
        log_text = item["log"]
        relevant_files = item["relevant_files"]
        error_type = item.get("error_type", "unknown")

        # Parse → embed → retrieve.
        parsed_log = log_parser.parse_log(log_text)
        log_embedding = log_embedder.embed_log(parsed_log)
        results = retriever.retrieve(
            log_embedding, parsed_log.query_text(), top_k=5
        )
        predictions = [r.file_path for r in results]

        # Compute per-query metrics.
        r1 = recall_at_k(predictions, relevant_files, k=1)
        r3 = recall_at_k(predictions, relevant_files, k=3)
        r5 = recall_at_k(predictions, relevant_files, k=5)
        m = mrr_score(predictions, relevant_files)

        r1_scores.append(r1)
        r3_scores.append(r3)
        r5_scores.append(r5)
        mrr_scores.append(m)

        by_type[error_type]["recall_at_1"].append(r1)
        by_type[error_type]["recall_at_3"].append(r3)
        by_type[error_type]["recall_at_5"].append(r5)
        by_type[error_type]["mrr"].append(m)

    # Aggregate.
    n = len(dataset) or 1  # guard against empty dataset
    per_error_type = {}
    for etype, metrics in by_type.items():
        per_error_type[etype] = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in metrics.items()
        }

    return EvalReport(
        recall_at_1=sum(r1_scores) / n,
        recall_at_3=sum(r3_scores) / n,
        recall_at_5=sum(r5_scores) / n,
        mrr=sum(mrr_scores) / n,
        num_samples=len(dataset),
        per_error_type=per_error_type,
    )
