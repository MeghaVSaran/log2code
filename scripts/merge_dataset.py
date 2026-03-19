"""
Dataset Merger — merge GitHub + synthetic pairs into train/dev splits.

Loads both data sources, deduplicates by id, filters out unknown error types,
applies stratified 80/20 train/dev split, and saves final dataset files.

Usage:
    python scripts/merge_dataset.py

Output:
    data/ground_truth/dataset.json  — full merged dataset with split field
    data/ground_truth/train.json    — train split only
    data/ground_truth/dev.json      — dev split only
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GITHUB_PATH = PROJECT_ROOT / "data" / "processed" / "github_pairs.json"
SYNTHETIC_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_errors.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ground_truth"

TRAIN_RATIO = 0.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> List[Dict]:
    """Load a JSON file, returning [] if missing or invalid."""
    if not path.exists():
        print(f"  [skip] {path} not found")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"  Loaded {len(data)} entries from {path.name}")
            return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] Could not load {path}: {e}")
        return []


def _save_json(data: List[Dict], path: Path) -> None:
    """Save list of dicts to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _stratified_split(
    entries: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[List[Dict], List[Dict]]:
    """Split entries into train/dev, stratified by error_type.

    Each error_type gets roughly the same train/dev ratio so both
    splits have proportional coverage of all categories.
    """
    rng = random.Random(seed)

    # Group by error_type
    by_type: dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        by_type[entry["error_type"]].append(entry)

    train: List[Dict] = []
    dev: List[Dict] = []

    for error_type, items in sorted(by_type.items()):
        rng.shuffle(items)
        split_idx = max(1, int(len(items) * train_ratio))
        # Ensure at least 1 in dev if we have >1 items
        if len(items) > 1 and split_idx == len(items):
            split_idx = len(items) - 1
        train.extend(items[:split_idx])
        dev.extend(items[split_idx:])

    # Shuffle within each split so they're not grouped by error_type
    rng.shuffle(train)
    rng.shuffle(dev)

    return train, dev


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data sources...")
    github_pairs = _load_json(GITHUB_PATH)
    synthetic_pairs = _load_json(SYNTHETIC_PATH)

    # Merge
    all_entries = github_pairs + synthetic_pairs
    print(f"\nTotal before dedup: {len(all_entries)}")

    # Deduplicate by id
    seen_ids: set = set()
    deduped: List[Dict] = []
    for entry in all_entries:
        eid = entry.get("id", "")
        if eid and eid not in seen_ids:
            seen_ids.add(eid)
            deduped.append(entry)
    print(f"After dedup: {len(deduped)}")

    # Filter out unknown error types
    filtered = [e for e in deduped if e.get("error_type") != "unknown"]
    n_unknown = len(deduped) - len(filtered)
    if n_unknown:
        print(f"Filtered out {n_unknown} entries with error_type='unknown'")
    print(f"After filtering: {len(filtered)}")

    if not filtered:
        print("No entries to process. Exiting.")
        return

    # Stratified train/dev split
    train, dev = _stratified_split(filtered, train_ratio=TRAIN_RATIO)

    # Add split field
    for entry in train:
        entry["split"] = "train"
    for entry in dev:
        entry["split"] = "dev"

    dataset = train + dev

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(dataset, OUTPUT_DIR / "dataset.json")
    _save_json(train, OUTPUT_DIR / "train.json")
    _save_json(dev, OUTPUT_DIR / "dev.json")

    # Summary
    print(f"\n{'='*50}")
    print(f"Dataset merge complete!")
    print(f"{'='*50}")
    print(f"Total:  {len(dataset)}")
    print(f"Train:  {len(train)}")
    print(f"Dev:    {len(dev)}")

    print(f"\nPer-error-type breakdown:")
    train_counts = Counter(e["error_type"] for e in train)
    dev_counts = Counter(e["error_type"] for e in dev)
    all_types = sorted(set(train_counts) | set(dev_counts))

    print(f"  {'Error Type':<25} {'Train':>6} {'Dev':>6} {'Total':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")
    for etype in all_types:
        t = train_counts.get(etype, 0)
        d = dev_counts.get(etype, 0)
        print(f"  {etype:<25} {t:>6} {d:>6} {t+d:>6}")

    print(f"\nPer-source breakdown:")
    source_counts = Counter(e.get("source", "unknown") for e in dataset)
    for src, count in source_counts.most_common():
        print(f"  {src}: {count}")

    print(f"\nSaved to:")
    print(f"  {OUTPUT_DIR / 'dataset.json'}")
    print(f"  {OUTPUT_DIR / 'train.json'}")
    print(f"  {OUTPUT_DIR / 'dev.json'}")


if __name__ == "__main__":
    main()
