"""
Vector Index — ChromaDB wrapper for dense retrieval.

See docs/2_system_architecture.md §5 for spec.
"""

from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "debugaid_code_chunks"


class IndexNotFoundError(Exception):
    pass


class VectorIndex:
    """Persistent ChromaDB vector index for code chunks."""

    def __init__(self, persist_dir: Path):
        """
        Args:
            persist_dir: Directory where ChromaDB stores its data.
                         Typically: repo_root/.debugaid/chroma/
        """
        raise NotImplementedError

    def build(self, chunks: List, embeddings: List) -> None:
        """Build or overwrite the index.

        Args:
            chunks: List of Chunk dataclass objects.
            embeddings: Parallel list of 768-dim vectors.
        """
        raise NotImplementedError

    def query(self, log_embedding: List[float], top_k: int = 20) -> List[Dict]:
        """Query index for nearest neighbors.

        Args:
            log_embedding: 768-dim query vector.
            top_k: Number of results to return.

        Returns:
            List of dicts: {chunk_id, file_path, function_name, start_line, score}
            Score is cosine similarity in [0, 1].

        Raises:
            IndexNotFoundError: If index has not been built.
        """
        raise NotImplementedError

    def exists(self) -> bool:
        """Return True if index has been built and can be queried."""
        raise NotImplementedError
