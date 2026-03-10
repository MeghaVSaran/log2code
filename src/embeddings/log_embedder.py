"""
Log Embedder — all-mpnet-base-v2 embeddings for parsed logs.

Outputs 768-dim vectors matching GraphCodeBERT output dimension.
See docs/2_system_architecture.md §4 for spec.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class LogEmbedder:
    """Generates 768-dim embeddings for log text."""

    def __init__(self, model_name: str = MODEL_NAME):
        raise NotImplementedError

    def embed_log(self, parsed_log) -> List[float]:
        """Embed a ParsedLog into a 768-dim vector.

        Input text = parsed_log.query_text()
        (error_message + space-joined identifiers)

        Args:
            parsed_log: ParsedLog dataclass instance.

        Returns:
            768-dim list of floats.
        """
        raise NotImplementedError

    def embed_text(self, text: str) -> List[float]:
        """Embed arbitrary text. Useful for testing."""
        raise NotImplementedError
