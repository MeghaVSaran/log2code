"""
Log Embedder — all-mpnet-base-v2 embeddings for parsed logs.

Outputs 768-dim vectors matching GraphCodeBERT output dimension.
See docs/2_system_architecture.md §4 for spec.
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class LogEmbedder:
    """Generates 768-dim embeddings for log text using all-mpnet-base-v2.

    The model is loaded lazily on first ``embed_*`` call so that simply
    importing this module does not trigger a model download.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """Initialise the embedder (model loaded lazily).

        Args:
            model_name: HuggingFace / sentence-transformers model identifier.
        """
        self._model_name = model_name
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the SentenceTransformer model if not already loaded."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info("Loading model %s …", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_log(self, parsed_log) -> List[float]:
        """Embed a ParsedLog into a 768-dim vector.

        Input text = ``parsed_log.query_text()``
        (error_message + space-joined identifiers).

        Args:
            parsed_log: ParsedLog dataclass instance.

        Returns:
            768-dim list of floats.
        """
        text = parsed_log.query_text()
        return self.embed_text(text)

    def embed_text(self, text: str) -> List[float]:
        """Embed arbitrary text. Useful for testing.

        Args:
            text: Arbitrary string to embed.

        Returns:
            768-dim list of floats.
        """
        self._ensure_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


# ----------------------------------------------------------------------
# Usage example
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from src.ingestion.log_parser import parse_log

    sample_log = "undefined reference to `Parser::resolveSymbol'"
    parsed = parse_log(sample_log)

    embedder = LogEmbedder()
    vector = embedder.embed_log(parsed)
    print(f"Embedding dim: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
