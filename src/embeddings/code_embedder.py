"""
Code Embedder — GraphCodeBERT embeddings for C++ function chunks.

See docs/2_system_architecture.md §3 for spec.
"""

from typing import List, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/graphcodebert-base"
DEFAULT_BATCH_SIZE = 16
MAX_LENGTH = 512  # GraphCodeBERT max token length


class CodeEmbedder:
    """Generates 768-dim embeddings for code chunks using GraphCodeBERT.

    The model is loaded lazily on first ``embed_*`` call so that simply
    importing this module does not trigger a model download.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        """Initialise the embedder (model loaded lazily).

        Args:
            model_name: HuggingFace model identifier.
            device: 'cpu' or 'cuda'.
        """
        self._model_name = model_name
        self._device = device
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the tokenizer and model if not already loaded."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModel
        import torch  # noqa: F401 — needed for .to(device)

        logger.info("Loading model %s on %s …", self._model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(
        self, chunks, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[np.ndarray]:
        """Embed a list of Chunk objects into vectors.

        Input format per chunk::

            "<function> {function_name} <context> {file_path}\\n{code_text}"

        Args:
            chunks: List of Chunk dataclass objects.
            batch_size: Number of chunks to embed per forward pass.

        Returns:
            List of 768-dim numpy arrays, one per chunk.
        """
        self._ensure_model()
        texts = [self._format_chunk(c) for c in chunks]
        embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_embs = self._embed_batch(batch)
            embeddings.extend(batch_embs)

        logger.debug("Embedded %d chunks.", len(embeddings))
        return embeddings

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string. Useful for testing.

        Args:
            text: Arbitrary text to embed.

        Returns:
            768-dim list of floats.
        """
        self._ensure_model()
        emb = self._embed_batch([text])[0]
        return emb.tolist()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _format_chunk(self, chunk) -> str:
        """Format a Chunk for model input."""
        return (
            f"<function> {chunk.function_name} "
            f"<context> {chunk.file_path}\n{chunk.code_text}"
        )

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Tokenize and run a forward pass on a batch, returning mean-pooled vectors."""
        import torch

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**encoded)

        # Mean pooling over the last hidden state, respecting attention mask.
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
        mask = encoded["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)
        summed = (last_hidden * mask).sum(dim=1)  # (batch, 768)
        counts = mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
        pooled = summed / counts  # (batch, 768)

        return [pooled[i].cpu().numpy() for i in range(pooled.size(0))]


# ----------------------------------------------------------------------
# Usage example
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from src.ingestion.code_parser import Chunk

    sample_chunk = Chunk(
        chunk_id="example.cpp::foo",
        file_path="example.cpp",
        function_name="foo",
        start_line=1,
        end_line=3,
        code_text="void foo() {\n    return;\n}",
    )

    embedder = CodeEmbedder()
    vectors = embedder.embed_chunks([sample_chunk])
    print(f"Embedding shape: {vectors[0].shape}")
    print(f"First 5 values: {vectors[0][:5]}")
