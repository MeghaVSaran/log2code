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
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """DebugAid — map C++ error logs to relevant source code."""
    pass


@cli.command()
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to C++ repository.")
@click.option("--force-reindex", is_flag=True, default=False, help="Rebuild index even if it exists.")
def index(repo, force_reindex):
    """Index a C++ repository for log-to-code retrieval."""
    raise NotImplementedError


@cli.command()
@click.option("--log", required=True, type=click.Path(exists=True), help="Path to log file.")
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to indexed C++ repository.")
@click.option("--top-k", default=5, show_default=True, help="Number of results to return.")
@click.option("--verbose", is_flag=True, default=False, help="Show scores and matched code snippets.")
@click.option("--output", default="text", type=click.Choice(["text", "json"]), help="Output format.")
def query(log, repo, top_k, verbose, output):
    """Query: given a log file, return the most likely source files."""
    raise NotImplementedError


@cli.command()
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Path to ground truth JSON.")
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to indexed C++ repository.")
def eval(dataset, repo):
    """Evaluate retrieval quality on a ground truth dataset."""
    raise NotImplementedError


@cli.command()
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to C++ repository.")
def info(repo):
    """Show index statistics for a repository."""
    raise NotImplementedError


if __name__ == "__main__":
    cli()
