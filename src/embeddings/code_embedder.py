"""
Code Embedder — GraphCodeBERT embeddings for C++ function chunks.

See docs/2_system_architecture.md §3 for spec.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/graphcodebert-base"
DEFAULT_BATCH_SIZE = 16
MAX_LENGTH = 512  # GraphCodeBERT max token length


class CodeEmbedder:
    """Generates 768-dim embeddings for code chunks using GraphCodeBERT."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        """
        Args:
            model_name: HuggingFace model identifier.
            device: 'cpu' or 'cuda'.
        """
        raise NotImplementedError

    def embed_chunks(self, chunks, batch_size: int = DEFAULT_BATCH_SIZE) -> List:
        """Embed a list of Chunk objects into vectors.

        Input format per chunk:
            "<function> {function_name} <context> {file_path}\\n{code_text}"

        Args:
            chunks: List of Chunk dataclass objects.
            batch_size: Number of chunks to embed per forward pass.

        Returns:
            List of 768-dim numpy arrays, one per chunk.
        """
        raise NotImplementedError

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string. Useful for testing."""
        raise NotImplementedError

    def _format_chunk(self, chunk) -> str:
        """Format a Chunk for model input."""
        return f"<function> {chunk.function_name} <context> {chunk.file_path}\n{chunk.code_text}"
