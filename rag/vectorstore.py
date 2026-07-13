"""
VectorStore - Manages the FAISS vector database and local embedding model.

Usage: From extensions.py or app.py, import VectorStore and instantiate it.
"""

import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


class VectorStore:
    """Manages the FAISS vector database and local embedding model."""

    def __init__(self, index_path: str, chunks_path: str, model_name: str, embedding_model: Optional[SentenceTransformer] = None):
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                "Run '01_build_vectorstore.py' first."
            )

        log.info("Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        log.info(f"FAISS index loaded: {self.index.ntotal} vectors")

        log.info("Loading chunk metadata...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[Dict] = json.load(f)
        log.info(f"Chunk metadata loaded: {len(self.chunks)} chunks")

        self.embedding_model_name = model_name
        # Allow injecting a pre-loaded embedding model (shared from extensions)
        self._model = embedding_model

        log.info(f"✓ Vector store ready — {self.index.ntotal} chunks, model={model_name}")

    def _get_model(self) -> SentenceTransformer:
        """Returns the SentenceTransformer instance (lazy-loaded if not injected)."""
        if self._model is None:
            log.info(f"Loading local embedding model: {self.embedding_model_name}")
            t0 = time.time()
            self._model = SentenceTransformer(self.embedding_model_name)
            log.info(
                f"✓ Model loaded in {time.time() - t0:.1f}s — "
                f"dim={self._model.get_sentence_embedding_dimension()}"
            )
        return self._model

    def _embed_text_sync(self, text: str) -> Optional[np.ndarray]:
        """
        Synchronously embeds a single text string using local SentenceTransformer.
        Returns a (1, dim) float32 numpy array normalized for cosine similarity,
        or None on failure.
        """
        try:
            model = self._get_model()
            embedding = model.encode(
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            return embedding
        except Exception as e:
            log.error(f"Local embedding failed: {e}")
            return None

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Returns the top-k most relevant chunks for the given query.
        E5 model requires the 'query:' prefix for retrieval queries.
        """
        query_text = f"query: {query}"

        # Embed using local model — fast, no API latency
        embedding = self._embed_text_sync(query_text)
        if embedding is None:
            log.error(f"Could not embed query: {query[:100]}")
            raise RuntimeError(
                "Sorgu embedding'i başarısız. Lütfen daha sonra tekrar deneyin."
            )

        scores, indices = self.index.search(embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:    # FAISS returns -1 for empty slots
                continue
            chunk = self.chunks[idx].copy()
            chunk["relevance_score"] = float(score)
            results.append(chunk)

        return results