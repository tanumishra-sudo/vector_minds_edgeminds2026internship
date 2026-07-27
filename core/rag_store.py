"""
Phase 2: Local RAG (Retrieval-Augmented Generation) Store.

This module provides a vector store backed by FAISS for efficient similarity
search over document chunks. Embeddings are computed using Sentence Transformers
with the 'all-MiniLM-L6-v2' model by default. The index lives in memory for
fast queries and can be persisted to / loaded from disk as a cache.

Typical workflow:
    1. Instantiate ``RAGStore``.
    2. Call ``add_documents`` with a list of chunk dicts (must contain a 'text' key).
    3. Call ``search`` with a natural-language query to retrieve the most relevant chunks.
    4. Optionally ``save_index`` / ``load_index`` to avoid re-embedding on restart.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
# SentenceTransformer is imported lazily inside __init__ to avoid
# RuntimeError('can't register atexit after shutdown') on Windows
# when Streamlit hot-reloads the module.

logger = logging.getLogger(__name__)


class RAGStore:
    """In-memory FAISS vector store with Sentence-Transformer embeddings.

    Attributes:
        model_name: Name of the Sentence Transformers model used for encoding.
        cache_dir:  Directory where the FAISS index and metadata are persisted.
        model:      The loaded ``SentenceTransformer`` instance.
        index:      The FAISS ``IndexFlatL2`` (or ``None`` before first add).
        documents:  Parallel list of chunk metadata aligned with index vectors.
    """

    # ------------------------------------------------------------------
    # Construction & initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "cache",
    ) -> None:
        """Initialise the embedding model and prepare the FAISS index.

        Args:
            model_name: HuggingFace Sentence Transformers model identifier.
                        Defaults to ``'all-MiniLM-L6-v2'`` (384-dim, fast).
            cache_dir:  Filesystem path for persisting the index and metadata.
                        The directory is created automatically if it does not
                        exist.
        """
        self.model_name: str = model_name
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        from sentence_transformers import SentenceTransformer
        logger.info("Loading Sentence Transformer model '%s' …", model_name)
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        logger.info("Model loaded successfully (dimension=%d).", self.model.get_sentence_embedding_dimension())

        self.index: faiss.IndexFlatL2 | None = None
        self.documents: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_index(self, dimension: int) -> None:
        """Create a fresh FAISS ``IndexFlatL2`` with the given vector dimension.

        This is called automatically the first time documents are added.  It
        can also be invoked manually to reset the index to a known dimension.

        Args:
            dimension: Dimensionality of the embedding vectors (e.g. 384 for
                       ``all-MiniLM-L6-v2``).
        """
        logger.debug("Initialising FAISS IndexFlatL2 with dimension=%d.", dimension)
        self.index = faiss.IndexFlatL2(dimension)

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def add_documents(self, chunks: list[dict[str, Any]]) -> None:
        """Encode document chunks and add them to the FAISS index.

        Each chunk **must** contain at least a ``'text'`` key whose value is the
        string to embed.  Any additional keys (``source``, ``page``,
        ``chunk_index``, …) are kept as metadata and returned verbatim during
        search.

        Args:
            chunks: List of dicts, each with at minimum a ``'text'`` key.

        Raises:
            ValueError: If *chunks* is empty or any chunk is missing ``'text'``.
        """
        if not chunks:
            raise ValueError("Cannot add an empty list of chunks.")

        texts: list[str] = []
        for i, chunk in enumerate(chunks):
            if "text" not in chunk:
                raise ValueError(f"Chunk at index {i} is missing the required 'text' key.")
            texts.append(chunk["text"])

        logger.info("Encoding %d chunk(s) with '%s' …", len(texts), self.model_name)
        embeddings: np.ndarray = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)

        # Lazily create the index on first call.
        if self.index is None:
            self._initialize_index(embeddings.shape[1])

        self.index.add(embeddings)  # type: ignore[union-attr]
        self.documents.extend(chunks)

        logger.info(
            "Added %d chunk(s). Total vectors in index: %d.",
            len(chunks),
            self.index.ntotal,  # type: ignore[union-attr]
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search the index for the chunks most similar to *query*.

        Args:
            query:  Natural-language search string.
            top_k:  Maximum number of results to return.  Clamped to the total
                    number of indexed documents if fewer are available.

        Returns:
            A list of dicts, each containing:
            - All original keys from the matching chunk (``text``, ``source``, …)
            - ``'score'``: The L2 distance (lower is more similar).
            - ``'rank'``: 1-based rank of the result.

            The list is ordered by ascending distance (best match first) and may
            be shorter than *top_k* if the index contains fewer vectors.

        Raises:
            RuntimeError: If the index is empty or has not been initialised.
        """
        if not self.has_documents():
            raise RuntimeError(
                "The RAG index is empty. Add documents before searching."
            )

        # Clamp top_k to available document count.
        effective_k: int = min(top_k, self.index.ntotal)  # type: ignore[union-attr]

        logger.debug("Searching index for top-%d matches …", effective_k)
        query_embedding: np.ndarray = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        distances, indices = self.index.search(query_embedding, effective_k)  # type: ignore[union-attr]

        results: list[dict[str, Any]] = []
        for rank, (dist, idx) in enumerate(
            zip(distances[0], indices[0]), start=1
        ):
            if idx == -1:
                # FAISS returns -1 for unfilled slots when k > ntotal.
                continue
            result = {**self.documents[int(idx)]}
            result["score"] = float(dist)
            result["rank"] = rank
            results.append(result)

        logger.info("Search returned %d result(s).", len(results))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, filename: str = "faiss_index") -> None:
        """Persist the FAISS index and document metadata to disk.

        Two files are written into ``self.cache_dir``:
        - ``<filename>.index`` – the raw FAISS binary index.
        - ``<filename>.meta``  – pickled list of document metadata dicts.

        Args:
            filename: Base name (without extension) for the output files.

        Raises:
            RuntimeError: If there is no index to save.
        """
        if not self.has_documents():
            raise RuntimeError("Nothing to save – the index is empty.")

        index_path: Path = self.cache_dir / f"{filename}.index"
        meta_path: Path = self.cache_dir / f"{filename}.meta"

        logger.info("Saving FAISS index to '%s' …", index_path)
        faiss.write_index(self.index, str(index_path))  # type: ignore[arg-type]

        logger.info("Saving metadata (%d docs) to '%s' …", len(self.documents), meta_path)
        with open(meta_path, "wb") as fh:
            pickle.dump(self.documents, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Index saved successfully.")

    def load_index(self, filename: str = "faiss_index") -> bool:
        """Load a previously saved FAISS index and metadata from disk.

        Args:
            filename: Base name (without extension) matching a prior
                      ``save_index`` call.

        Returns:
            ``True`` if the index was loaded successfully, ``False`` if the
            cache files do not exist or an error occurred.
        """
        index_path: Path = self.cache_dir / f"{filename}.index"
        meta_path: Path = self.cache_dir / f"{filename}.meta"

        if not index_path.exists() or not meta_path.exists():
            logger.warning(
                "Cache files not found ('%s', '%s'). Nothing to load.",
                index_path,
                meta_path,
            )
            return False

        try:
            logger.info("Loading FAISS index from '%s' …", index_path)
            self.index = faiss.read_index(str(index_path))

            logger.info("Loading metadata from '%s' …", meta_path)
            with open(meta_path, "rb") as fh:
                self.documents = pickle.load(fh)  # noqa: S301

            logger.info(
                "Index loaded successfully (%d vectors, %d metadata entries).",
                self.index.ntotal,
                len(self.documents),
            )
            return True
        except Exception:
            logger.exception("Failed to load index from cache.")
            return False

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear the FAISS index and all stored metadata.

        After calling this the store behaves as if freshly constructed (but the
        embedding model stays loaded).
        """
        self.index = None
        self.documents = []
        logger.info("RAG store cleared.")

    def has_documents(self) -> bool:
        """Check whether any documents have been indexed.

        Returns:
            ``True`` if the index exists **and** contains at least one vector.
        """
        return self.index is not None and self.index.ntotal > 0

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        count = self.index.ntotal if self.index is not None else 0
        return (
            f"RAGStore(model={self.model_name!r}, "
            f"vectors={count}, "
            f"cache_dir={str(self.cache_dir)!r})"
        )

    def __len__(self) -> int:
        """Return the number of vectors currently in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
