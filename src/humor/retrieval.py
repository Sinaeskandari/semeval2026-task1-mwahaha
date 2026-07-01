"""Dense Wikipedia retriever for the RAG generation variant.

A subset of a pre-embedded Wikipedia dump is loaded into memory and searched
with exact cosine similarity. Because the document embeddings are pre-computed,
only the (short) query needs to be encoded at run time, which keeps retrieval
cheap even on CPU.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise a matrix so dot products equal cosine similarities."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-12)


class WikipediaRetriever:
    """Cosine-similarity retriever over a pre-embedded Wikipedia subset.

    Args:
        n_docs: Number of Wikipedia documents to load. Larger corpora improve
            recall at the cost of memory and load time (the paper sweeps
            5k-25k; see :class:`humor.config.RagConfig`).
        embedding_model: Sentence-Transformers model used to encode queries.
            Must match the space of the pre-computed document embeddings.
        dataset: Hugging Face dataset id providing ``text`` and ``embeddings``.
        revision: Dataset revision/branch holding the embedded split.
        device: Device for the query encoder (``"cpu"`` is intentionally the
            default: it avoids competing with the 4-bit LLM for GPU memory).
    """

    def __init__(
        self,
        n_docs: int = 25_000,
        *,
        embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
        dataset: str = "not-lain/wikipedia",
        revision: str = "embedded",
        device: str = "cpu",
    ) -> None:
        # Imported lazily to keep the package importable without the RAG extras.
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer

        logger.info("Loading %d embedded Wikipedia documents ...", n_docs)
        data = load_dataset(dataset, revision=revision, split=f"train[:{n_docs}]")
        self.texts: list[str] = [str(x) for x in data["text"]]
        self.embeddings = _l2_normalise(np.asarray(data["embeddings"], dtype=np.float32))

        logger.info("Loading query encoder '%s' on %s ...", embedding_model, device)
        self.encoder = SentenceTransformer(embedding_model, device=device)
        logger.info("Retriever ready (%d docs).", len(self.texts))

    def retrieve(self, query: str, k: int = 4, max_chars: int = 1_200) -> str:
        """Return the top-``k`` documents for ``query`` as a single string.

        Args:
            query: Natural-language retrieval query.
            k: Number of documents to concatenate.
            max_chars: Hard cap on the returned context length (keeps the
                prompt within the model's context window).

        Returns:
            The concatenated top-``k`` document texts, truncated to
            ``max_chars``. Returns an empty string on any failure so generation
            can proceed without context.
        """
        try:
            q = self.encoder.encode([query]).astype(np.float32)
            q = _l2_normalise(q)
            sims = self.embeddings @ q[0]
            top_idx = np.argsort(-sims)[:k]
            combined = "\n\n".join(self.texts[i] for i in top_idx)
            return combined[:max_chars]
        except Exception as exc:  # noqa: BLE001 - retrieval must never crash the run
            logger.warning("Retrieval failed for query %r: %s", query[:60], exc)
            return ""
