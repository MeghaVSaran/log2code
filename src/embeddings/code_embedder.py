"""
Code Embedder — configurable embeddings for C++ function chunks.

Supports two backends:
  - ``graphcodebert``: microsoft/graphcodebert-base (original, 125M params)
  - ``mpnet``: sentence-transformers/all-mpnet-base-v2 (shared space with
    LogEmbedder — required for valid dense retrieval)

See docs/2_system_architecture.md §3 for spec.
"""

from typing import List, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)

GRAPHCODEBERT_MODEL = "microsoft/graphcodebert-base"
MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_BATCH_SIZE = 16
MAX_LENGTH = 512  # GraphCodeBERT max token length


class CodeEmbedder:
    """Generates 768-dim embeddings for code chunks.

    Supports two backends:

    - ``"graphcodebert"`` — uses ``microsoft/graphcodebert-base`` via
      HuggingFace transformers (original behaviour).  Produces vectors
      in a *different* space than LogEmbedder, so dense cosine
      similarity between code and log embeddings is **not valid**.
    - ``"mpnet"`` — uses ``sentence-transformers/all-mpnet-base-v2``,
      the same model as LogEmbedder.  Dense similarity between code
      and log embeddings is valid in a common vector space.  This is
      a practical MVP fix, not necessarily the optimal final code model.

    The model is loaded lazily on first ``embed_*`` call.
    """

    def __init__(
        self,
        backend: str = "mpnet",
        device: str = "cpu",
    ):
        """Initialise the embedder (model loaded lazily).

        Args:
            backend: ``"graphcodebert"`` or ``"mpnet"``.
            device: ``"cpu"`` or ``"cuda"``.
        """
        if backend not in ("graphcodebert", "mpnet"):
            raise ValueError(
                f"Unknown backend '{backend}'. Use 'graphcodebert' or 'mpnet'."
            )
        self._backend = backend
        self._device = device
        self._tokenizer = None
        self._model = None

    @property
    def embedding_backend(self) -> str:
        """Return the backend name for index metadata."""
        return self._backend

    @property
    def model_name(self) -> str:
        """Return the HuggingFace model name."""
        if self._backend == "graphcodebert":
            return GRAPHCODEBERT_MODEL
        return MPNET_MODEL

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the model if not already loaded."""
        if self._model is not None:
            return

        if self._backend == "graphcodebert":
            self._load_graphcodebert()
        else:
            self._load_mpnet()

    def _load_graphcodebert(self) -> None:
        from transformers import AutoTokenizer, AutoModel
        import torch  # noqa: F401

        logger.info("Loading GraphCodeBERT on %s …", self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(GRAPHCODEBERT_MODEL)
        self._model = AutoModel.from_pretrained(GRAPHCODEBERT_MODEL).to(self._device)
        self._model.eval()
        logger.info("GraphCodeBERT loaded.")

    def _load_mpnet(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading mpnet on %s …", self._device)
        self._model = SentenceTransformer(MPNET_MODEL, device=self._device)
        logger.info("mpnet loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(
        self, chunks, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[np.ndarray]:
        """Embed a list of Chunk objects into vectors.

        Args:
            chunks: List of Chunk dataclass objects.
            batch_size: Number of chunks to embed per forward pass.

        Returns:
            List of 768-dim numpy arrays, one per chunk.
        """
        self._ensure_model()
        texts = [self._format_chunk(c) for c in chunks]

        if self._backend == "mpnet":
            return self._embed_mpnet(texts, batch_size)
        return self._embed_graphcodebert(texts, batch_size)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string. Useful for testing.

        Args:
            text: Arbitrary text to embed.

        Returns:
            768-dim list of floats.
        """
        self._ensure_model()
        if self._backend == "mpnet":
            embs = self._embed_mpnet([text])
            return embs[0].tolist()
        embs = self._embed_graphcodebert([text])
        return embs[0].tolist()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _format_chunk(self, chunk) -> str:
        """Format a Chunk for model input."""
        return (
            f"{chunk.function_name} {chunk.file_path}\n{chunk.code_text}"
        )

    def _embed_mpnet(
        self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[np.ndarray]:
        """Embed using SentenceTransformer (mpnet)."""
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [embeddings[i] for i in range(len(texts))]

    def _embed_graphcodebert(
        self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[np.ndarray]:
        """Embed using GraphCodeBERT (HF transformers)."""
        import torch

        all_embs: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**encoded)

            last_hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts

            all_embs.extend(
                [pooled[i].cpu().numpy() for i in range(pooled.size(0))]
            )

        return all_embs


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

    embedder = CodeEmbedder(backend="mpnet")
    vectors = embedder.embed_chunks([sample_chunk])
    print(f"Backend: {embedder.embedding_backend}")
    print(f"Embedding shape: {vectors[0].shape}")
    print(f"First 5 values: {vectors[0][:5]}")

