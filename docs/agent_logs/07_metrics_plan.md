# Evaluation Metrics — Implementation Plan

## Goal
Implement `src/evaluation/metrics.py` with Recall@K, MRR, and evaluate_dataset.

## Changes

### [MODIFY] metrics.py (was stub)
- `recall_at_k()`: set intersection on predictions[:k] vs ground_truth
- `mrr_score()`: find first hit in predictions, return 1/rank
- `evaluate_dataset()`: parse→embed→retrieve for each item, aggregate means, per-error-type breakdown
- Updated signature to accept `log_parser` and `log_embedder` params

### [NEW] test_metrics.py
18 tests: recall (8), mrr (6), evaluate_dataset (4 with fake objects)
