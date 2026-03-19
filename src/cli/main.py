"""
DebugAid CLI — main entry point.

Commands:
  debugaid index  --repo PATH [--force-reindex]
  debugaid query  --log PATH --repo PATH [--top-k N] [--verbose] [--output json|text]
  debugaid eval   --dataset PATH --repo PATH
  debugaid info   --repo PATH

See docs/5_user_stories.md for expected behavior.
"""

import click
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEBUGAID_DIR = ".debugaid"
CHROMA_DIR = "chroma"
BM25_FILE = "bm25.pkl"


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------

@click.group()
def cli():
    """DebugAid — map C++ error logs to relevant source code."""
    pass


# ------------------------------------------------------------------
# INDEX command
# ------------------------------------------------------------------

@cli.command()
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to C++ repository.")
@click.option("--force-reindex", is_flag=True, default=False, help="Rebuild index even if it exists.")
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Device for embedding.")
@click.option("--include-tests", is_flag=True, default=False, help="Include test/benchmark files in index.")
def index(repo, force_reindex, device, include_tests):
    """Index a C++ repository for log-to-code retrieval."""
    try:
        import torch

        # Auto-detect / validate device.
        if device == "cuda":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                click.echo(f"Using device: cuda ({gpu_name})")
            else:
                click.echo("Warning: CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
        if device == "cpu":
            click.echo("Using device: cpu")

        repo_path = Path(repo).resolve()
        debugaid_path = repo_path / DEBUGAID_DIR

        # Check for existing index.
        if debugaid_path.exists() and not force_reindex:
            click.echo("Index exists. Use --force-reindex to rebuild.")
            sys.exit(0)

        click.echo(f"Indexing repository: {repo_path}")
        t_start = time.time()

        # 1. Parse repository.
        from src.ingestion.code_parser import parse_repository

        click.echo("Parsing C++ files...")
        if not include_tests:
            click.echo("  (excluding test/benchmark files; use --include-tests to keep them)")
        chunks = parse_repository(repo_path, include_tests=include_tests)
        if not chunks:
            click.echo("No C++ files found in repository.", err=True)
            sys.exit(1)
        click.echo(f"  Found {len(chunks)} functions in {repo_path}")

        # 2. Embed chunks.
        from src.embeddings.code_embedder import CodeEmbedder

        click.echo("Embedding code chunks (this may take a while)...")
        embedder = CodeEmbedder(device=device)
        embeddings = embedder.embed_chunks(chunks)
        click.echo(f"  Embedded {len(embeddings)} chunks")

        # 3. Build vector index.
        from src.indexing.vector_index import VectorIndex

        debugaid_path.mkdir(parents=True, exist_ok=True)
        chroma_path = debugaid_path / CHROMA_DIR

        click.echo("Building vector index...")
        vector_index = VectorIndex(chroma_path)
        vector_index.build(chunks, embeddings)

        # 4. Build BM25 index.
        from src.indexing.bm25_index import BM25Index

        click.echo("Building BM25 index...")
        bm25_index = BM25Index()
        bm25_index.build(chunks)
        bm25_index.save(debugaid_path / BM25_FILE)

        elapsed = time.time() - t_start
        click.echo(f"Indexed {len(chunks)} chunks in {elapsed:.1f} seconds")

    except Exception as exc:
        click.echo(f"Error during indexing: {exc}", err=True)
        sys.exit(1)


# ------------------------------------------------------------------
# QUERY command
# ------------------------------------------------------------------

@cli.command()
@click.option("--log", "log_path", required=True, type=click.Path(exists=True), help="Path to log file.")
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to indexed C++ repository.")
@click.option("--top-k", default=5, show_default=True, help="Number of results to return.")
@click.option("--verbose", is_flag=True, default=False, help="Show scores and matched code snippets.")
@click.option("--output", default="text", type=click.Choice(["text", "json"]), help="Output format.")
def query(log_path, repo, top_k, verbose, output):
    """Query: given a log file, return the most likely source files."""
    try:
        repo_path = Path(repo).resolve()
        debugaid_path = repo_path / DEBUGAID_DIR

        if not debugaid_path.exists():
            click.echo("No index found. Run 'debugaid index --repo .' first.", err=True)
            sys.exit(1)

        # 1. Read log file.
        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")

        # 2. Parse log.
        from src.ingestion.log_parser import parse_log, extract_source_paths

        parsed_log = parse_log(log_text)
        source_paths = extract_source_paths(log_text, repo_root=repo_path)

        # 3. Load indices.
        from src.indexing.vector_index import VectorIndex
        from src.indexing.bm25_index import BM25Index

        vector_index = VectorIndex(debugaid_path / CHROMA_DIR)
        bm25_index = BM25Index()
        bm25_index.load(debugaid_path / BM25_FILE)

        # 4. Embed log.
        from src.embeddings.log_embedder import LogEmbedder

        log_embedder = LogEmbedder()
        log_embedding = log_embedder.embed_log(parsed_log)

        # 5. Retrieve.
        from src.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(vector_index, bm25_index)
        results = retriever.retrieve(
            log_embedding,
            parsed_log.query_text(),
            top_k=top_k,
            source_paths=source_paths,
        )

        if not results:
            click.echo("No results found.", err=True)
            sys.exit(1)

        # 6. Output.
        if output == "json":
            json_results = [
                {
                    "rank": r.rank,
                    "file_path": r.file_path,
                    "function_name": r.function_name,
                    "start_line": r.start_line,
                    "score": round(r.score, 4),
                    "dense_score": round(r.dense_score, 4),
                    "bm25_score": round(r.bm25_score, 4),
                }
                for r in results
            ]
            click.echo(json.dumps(json_results, indent=2))
        else:
            click.echo(f"\nError type: {parsed_log.error_type}")
            click.echo(f"Query: {parsed_log.query_text()[:120]}")
            click.echo(f"\nTop {len(results)} results:\n")

            for r in results:
                click.echo(f"  #{r.rank}  {r.file_path}:{r.start_line}")
                click.echo(f"       Function: {r.function_name}")
                if verbose:
                    click.echo(f"       Score: {r.score:.4f} "
                               f"(dense={r.dense_score:.4f}, bm25={r.bm25_score:.4f})")
                click.echo()

    except Exception as exc:
        click.echo(f"Error during query: {exc}", err=True)
        sys.exit(1)


# ------------------------------------------------------------------
# EVAL command
# ------------------------------------------------------------------

@cli.command("eval")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Path to ground truth JSON.")
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to indexed C++ repository.")
@click.option("--repo-filter", default=None, help="Only eval samples whose relevant_files all start with this prefix (e.g. 'absl/').")
def eval_cmd(dataset, repo, repo_filter):
    """Evaluate retrieval quality on a ground truth dataset."""
    try:
        repo_path = Path(repo).resolve()
        debugaid_path = repo_path / DEBUGAID_DIR

        if not debugaid_path.exists():
            click.echo("No index found. Run 'debugaid index --repo .' first.", err=True)
            sys.exit(1)

        # 1. Load dataset.
        with open(dataset, "r", encoding="utf-8") as f:
            data = json.load(f)
        click.echo(f"Loaded {len(data)} samples from {dataset}")

        # Apply optional repo filter to drop samples from unindexed repos.
        if repo_filter:
            original_count = len(data)
            data = [
                s for s in data
                if all(f.startswith(repo_filter) for f in s.get("relevant_files", []))
            ]
            click.echo(
                f"Filtered to {len(data)}/{original_count} samples "
                f"matching prefix '{repo_filter}'"
            )
            if not data:
                click.echo("No samples remain after filtering.", err=True)
                sys.exit(1)

        # 2. Load indices.
        from src.indexing.vector_index import VectorIndex
        from src.indexing.bm25_index import BM25Index

        vector_index = VectorIndex(debugaid_path / CHROMA_DIR)
        bm25_index = BM25Index()
        bm25_index.load(debugaid_path / BM25_FILE)

        # 3. Set up pipeline.
        from src.ingestion.log_parser import parse_log
        from src.embeddings.log_embedder import LogEmbedder
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.evaluation.metrics import evaluate_dataset

        log_embedder = LogEmbedder()
        retriever = HybridRetriever(vector_index, bm25_index)

        # Create a minimal log_parser wrapper that has .parse_log().
        class _LogParserWrapper:
            @staticmethod
            def parse_log(text):
                return parse_log(text)

        click.echo("Running evaluation...")
        t_start = time.time()
        report = evaluate_dataset(data, retriever, _LogParserWrapper(), log_embedder, repo_root=repo_path)
        elapsed = time.time() - t_start

        # 4. Print report.
        filter_label = f" [filtered: {repo_filter}]" if repo_filter else ""
        click.echo(f"\n{'='*60}")
        click.echo(f"  Evaluation Report  ({report.num_samples} samples){filter_label}")
        click.echo(f"{'='*60}")
        if repo_filter:
            click.echo(f"  NOTE: Results are for the {repo_filter}-filtered subset only.")
            click.echo(f"        Do NOT report as overall system performance.")
        click.echo(f"  Recall@1:  {report.recall_at_1:.4f}")
        click.echo(f"  Recall@3:  {report.recall_at_3:.4f}")
        click.echo(f"  Recall@5:  {report.recall_at_5:.4f}")
        click.echo(f"  MRR:       {report.mrr:.4f}")
        click.echo(f"{'='*60}")

        if report.per_error_type:
            click.echo(f"\n  Per Error Type:")
            click.echo(f"  {'Type':<20} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}")
            click.echo(f"  {'-'*44}")
            for etype, metrics in sorted(report.per_error_type.items()):
                click.echo(
                    f"  {etype:<20} "
                    f"{metrics.get('recall_at_1', 0):.4f} "
                    f"{metrics.get('recall_at_3', 0):.4f} "
                    f"{metrics.get('recall_at_5', 0):.4f} "
                    f"{metrics.get('mrr', 0):.4f}"
                )

        click.echo(f"\n  Completed in {elapsed:.1f} seconds")

    except Exception as exc:
        click.echo(f"Error during evaluation: {exc}", err=True)
        sys.exit(1)


# ------------------------------------------------------------------
# INFO command
# ------------------------------------------------------------------

@cli.command()
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to C++ repository.")
def info(repo):
    """Show index statistics for a repository."""
    try:
        repo_path = Path(repo).resolve()
        debugaid_path = repo_path / DEBUGAID_DIR

        if not debugaid_path.exists():
            click.echo("No index found. Run 'debugaid index --repo .' first.", err=True)
            sys.exit(1)

        click.echo(f"\nRepository:  {repo_path}")

        # Chunk count from ChromaDB.
        chroma_path = debugaid_path / CHROMA_DIR
        chunk_count = "unknown"
        if chroma_path.exists():
            try:
                from src.indexing.vector_index import VectorIndex, COLLECTION_NAME
                vi = VectorIndex(chroma_path)
                col = vi._client.get_collection(COLLECTION_NAME)
                chunk_count = col.count()
            except Exception:
                pass
        click.echo(f"Chunks:      {chunk_count}")

        # Index size on disk.
        total_size = sum(
            f.stat().st_size for f in debugaid_path.rglob("*") if f.is_file()
        )
        if total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        else:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        click.echo(f"Index size:  {size_str}")

        # Date built (modification time of BM25 file).
        bm25_path = debugaid_path / BM25_FILE
        if bm25_path.exists():
            mtime = bm25_path.stat().st_mtime
            date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            click.echo(f"Date built:  {date_str}")
        else:
            click.echo("Date built:  unknown")

        click.echo()

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
