"""
Synthetic Error Generator — inject errors into C++ files, compile, capture logs.

Programmatically introduces common C++ errors into real source files,
compiles them, captures the error log, and saves labeled pairs.
Ground truth = the file that was modified.

Usage:
    python scripts/generate_synthetic_errors.py --repo /path/to/cpp/repo

See docs/4_dataset_strategy.md §2 for the full strategy.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c"}

# Same patterns used by code_parser.py to skip test / benchmark files.
EXCLUDE_PATTERNS = [
    "_test.cc", "_test.cpp", "_test.h",
    "test_", "tests/", "/test/",
    "_unittest.cc", "_benchmark.cc",
    "_bench.cc", "benchmark/",
]

# Error types we can inject.
INJECTION_TYPES = [
    "linker_error",
    "include_error",
    "compiler_error",      # undeclared identifier
    # "multiple_definition",  # needs two-file compile, skip for now
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_test_file(rel_path: str) -> bool:
    """Return True if rel_path matches any test/benchmark pattern."""
    lower = rel_path.lower().replace("\\", "/")
    return any(pat in lower for pat in EXCLUDE_PATTERNS)


def _collect_cpp_files(repo_path: Path) -> List[Path]:
    """Return all non-test .cpp/.cc files in the repo."""
    files = []
    for ext in CPP_EXTENSIONS:
        for f in repo_path.rglob(f"*{ext}"):
            rel = str(f.relative_to(repo_path)).replace("\\", "/")
            if not _is_test_file(rel):
                files.append(f)
    return files


def _compile(file_path: Path, repo_root: Path, extra_files: List[Path] = None) -> str:
    """Compile a single C++ file and return stderr output.

    Uses: g++ -c {file} -I{repo_root} -std=c++17 -fsyntax-only 2>&1
    """
    cmd = ["g++", "-c", str(file_path), f"-I{repo_root}", "-std=c++17",
           "-fsyntax-only"]
    if extra_files:
        cmd.extend(str(f) for f in extra_files)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stderr
    except subprocess.TimeoutExpired:
        return ""
    except FileNotFoundError:
        logger.error("g++ not found — make sure it's installed and on PATH.")
        sys.exit(1)


def _extract_error_lines(stderr: str, max_lines: int = 15) -> str:
    """Extract the first max_lines of meaningful error output."""
    lines = stderr.strip().splitlines()
    # Keep only lines that look like errors/warnings
    useful = [l for l in lines if "error" in l.lower() or "undefined" in l.lower()
              or "fatal" in l.lower() or "no such" in l.lower()
              or "undeclared" in l.lower() or "no matching" in l.lower()]
    if not useful:
        useful = lines[:max_lines]
    return "\n".join(useful[:max_lines])


# ---------------------------------------------------------------------------
# Error injection functions
# ---------------------------------------------------------------------------

def _inject_linker_error(file_path: Path) -> Optional[str]:
    """Comment out the first function body, keeping the signature.

    Returns the function name if successful, None otherwise.
    """
    content = file_path.read_text(encoding="utf-8", errors="replace")
    # Find a function definition: returntype name(args) { ... }
    # Simple heuristic: look for lines that end with '{' after a ')'
    pattern = re.compile(
        r"^([^\n]*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?)\{",
        re.MULTILINE,
    )
    match = pattern.search(content)
    if not match:
        return None

    # Find the matching closing brace
    start = match.end() - 1  # position of '{'
    depth = 0
    end = start
    for i in range(start, len(content)):
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    else:
        return None

    # Extract function name from signature
    sig = match.group(1).strip()
    name_match = re.search(r'(\w+)\s*\(', sig)
    if not name_match:
        return None
    func_name = name_match.group(1)

    # Comment out the body (keep signature as declaration + semicolon)
    new_content = content[:match.end() - 1] + ";\n/* body removed */\n" + content[end + 1:]
    file_path.write_text(new_content, encoding="utf-8")
    return func_name


def _inject_include_error(file_path: Path) -> Optional[str]:
    """Remove the first non-standard #include line.

    Returns the include filename if successful.
    """
    content = file_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines(keepends=True)

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match #include "..." or #include <...> but skip standard library
        m = re.match(r'#include\s+[<"]([^>"]+)[>"]', stripped)
        if m:
            include_name = m.group(1)
            # Skip standard headers (no dot = likely standard)
            if '.' in include_name or '/' in include_name:
                lines[i] = f"// REMOVED: {line}"
                file_path.write_text("".join(lines), encoding="utf-8")
                return include_name
    return None


def _inject_undeclared_identifier(file_path: Path) -> Optional[str]:
    """Rename a function call to produce 'use of undeclared identifier'.

    Returns the original function name if successful.
    """
    content = file_path.read_text(encoding="utf-8", errors="replace")

    # Find function calls: identifier(
    calls = list(re.finditer(r'\b([a-zA-Z_]\w{3,})\s*\(', content))
    if not calls:
        return None

    # Pick a call in the middle of the file (not near the top where
    # declarations live)
    mid = len(calls) // 2
    match = calls[mid]
    orig_name = match.group(1)

    # Skip common keywords and very short names
    keywords = {"return", "while", "sizeof", "static_cast", "dynamic_cast",
                "reinterpret_cast", "const_cast", "decltype", "alignof",
                "noexcept", "throw", "catch", "switch", "assert", "define",
                "include", "ifdef", "ifndef", "endif", "elif", "pragma",
                "error", "warning"}
    if orig_name.lower() in keywords:
        return None

    # Rename just this one call site
    mangled = f"__BOGUS_{orig_name}"
    start, end = match.start(1), match.end(1)
    new_content = content[:start] + mangled + content[end:]
    file_path.write_text(new_content, encoding="utf-8")
    return orig_name


# ---------------------------------------------------------------------------
# Core injection + compile function
# ---------------------------------------------------------------------------

def inject_and_compile(
    file_path: Path,
    repo_root: Path,
    error_type: str,
) -> Optional[Dict[str, Any]]:
    """Inject an error, compile, capture log, restore file.

    Args:
        file_path: Absolute path to the C++ source file.
        repo_root: Root of the repository (used for -I flag).
        error_type: One of INJECTION_TYPES.

    Returns:
        Dict with {log, relevant_files, error_type, source} or None
        if compile unexpectedly succeeded or injection failed.
    """
    # Make a backup before anything
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    shutil.copy2(file_path, backup_path)

    try:
        # Inject the error
        if error_type == "linker_error":
            result = _inject_linker_error(file_path)
        elif error_type == "include_error":
            result = _inject_include_error(file_path)
        elif error_type == "compiler_error":
            result = _inject_undeclared_identifier(file_path)
        else:
            return None

        if result is None:
            return None

        # Compile and capture stderr
        stderr = _compile(file_path, repo_root)

        if not stderr or not stderr.strip():
            # Compile succeeded — no error produced
            return None

        log_snippet = _extract_error_lines(stderr)
        if not log_snippet or len(log_snippet) < 10:
            return None

        rel_path = str(file_path.relative_to(repo_root)).replace("\\", "/")

        return {
            "id": f"synthetic-{error_type}-{rel_path.replace('/', '_')}-{uuid.uuid4().hex[:8]}",
            "log": log_snippet,
            "relevant_files": [rel_path],
            "error_type": error_type,
            "source": "synthetic",
        }

    except Exception as exc:
        logger.debug("Injection failed for %s: %s", file_path, exc)
        return None

    finally:
        # ALWAYS restore the original file
        if backup_path.exists():
            shutil.copy2(backup_path, file_path)
            backup_path.unlink()


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_synthetic_errors(
    repo_path: Path,
    target_count: int = 200,
) -> List[Dict]:
    """Generate synthetic error pairs by injecting errors into C++ files.

    Args:
        repo_path: Path to the C++ repository.
        target_count: Stop after collecting this many pairs.

    Returns:
        List of result dicts.
    """
    repo_path = Path(repo_path).resolve()
    output_path = RAW_DIR / "synthetic_errors.json"

    # Load existing results for incremental support
    results: List[Dict] = []
    seen_ids: set = set()
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                seen_ids = {r["id"] for r in results}
                logger.info("Loaded %d existing results", len(results))
        except (json.JSONDecodeError, OSError):
            pass

    if len(results) >= target_count:
        print(f"Already have {len(results)} pairs (target={target_count}). Done.")
        return results

    # Collect candidate files
    cpp_files = _collect_cpp_files(repo_path)
    random.shuffle(cpp_files)
    print(f"Found {len(cpp_files)} non-test C++ files in {repo_path}")

    pbar = tqdm(total=target_count - len(results), desc="Generating pairs")

    for file_path in cpp_files:
        if len(results) >= target_count:
            break

        for error_type in INJECTION_TYPES:
            if len(results) >= target_count:
                break

            pair = inject_and_compile(file_path, repo_path, error_type)
            if pair is None:
                continue

            if pair["id"] in seen_ids:
                continue

            results.append(pair)
            seen_ids.add(pair["id"])
            pbar.update(1)

            # Save incrementally every 10 pairs
            if len(results) % 10 == 0:
                _save_results(results, output_path)

    pbar.close()

    # Final save
    _save_results(results, output_path)
    return results


def _save_results(results: List[Dict], path: Path) -> None:
    """Atomically write results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Run synthetic error generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic C++ error pairs for evaluation.",
    )
    parser.add_argument(
        "--repo", type=str, default="/tmp/abseil",
        help="Path to a C++ repository (default: /tmp/abseil)",
    )
    parser.add_argument(
        "--count", type=int, default=200,
        help="Target number of pairs to generate (default: 200)",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo)
    if not repo_path.exists():
        print(f"Error: repo path {repo_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    results = generate_synthetic_errors(repo_path, target_count=args.count)

    # Summary
    from collections import Counter
    type_counts = Counter(r["error_type"] for r in results)

    print(f"\n{'='*50}")
    print(f"Synthetic generation complete!")
    print(f"Total pairs: {len(results)}")
    print(f"Output: {RAW_DIR / 'synthetic_errors.json'}")
    print(f"{'='*50}")
    print("\nPer error type:")
    for etype, count in type_counts.most_common():
        print(f"  {etype}: {count}")


if __name__ == "__main__":
    main()
