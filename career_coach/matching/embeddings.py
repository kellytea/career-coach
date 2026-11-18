"""Text embedders and a vector index (FAISS when installed, exact NumPy otherwise)."""

import hashlib
import re
from typing import Protocol

import numpy as np

from .text import normalize


MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MPNET = "sentence-transformers/all-mpnet-base-v2"


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalized vectors."""


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    """Sentence-BERT family encoder (MiniLM, MPNet). Needs ``sentence-transformers``."""

    def __init__(self, model_name: str = MINILM, *, batch_size: int = 64, device: str | None = None):
        self.name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ImportError("Install the matching extra: uv sync --extra matching") from error
            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._load().encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)
        return _normalize_rows(vectors)


class HashingEmbedder:
    """Deterministic word + character n-gram hashing embedder.

    For offline tests and pipeline checks only: it captures spelling overlap,
    not meaning, so it must not be reported as a Sentence-BERT result.
    """

    def __init__(self, dim: int = 512, name: str = "hashing"):
        self.dim = dim
        self.name = name

    def _features(self, text: str):
        words = re.findall(r"[a-z0-9+#.]+", normalize(text))
        for word in words:
            yield "w:" + word, 1.0
            padded = f"<{word}>"
            for n in (3, 4):
                for i in range(len(padded) - n + 1):
                    yield f"c{n}:" + padded[i:i + n], 0.5

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign * weight
        return _normalize_rows(matrix)


class VectorIndex:
    """Inner-product index over normalized vectors (cosine similarity)."""

    def __init__(self, vectors: np.ndarray, *, use_faiss: bool | None = None):
        self.vectors = _normalize_rows(vectors) if len(vectors) else np.asarray(vectors, dtype=np.float32)
        self._faiss = None
        if use_faiss is not False and len(self.vectors):
            try:
                import faiss
            except ImportError:
                if use_faiss:
                    raise
            else:
                # Exact search: the corpus (~100k chunks) does not need approximate indexes.
                index = faiss.IndexFlatIP(self.vectors.shape[1])
                index.add(np.ascontiguousarray(self.vectors))
                self._faiss = index

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def __len__(self):
        return len(self.vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, ids), each of shape (n_queries, min(k, size)), best first."""
        k = min(k, len(self))
        queries = _normalize_rows(np.atleast_2d(queries))
        if k == 0:
            return np.zeros((len(queries), 0), np.float32), np.zeros((len(queries), 0), np.int64)
        if self._faiss is not None:
            scores, ids = self._faiss.search(np.ascontiguousarray(queries), k)
            return scores, ids
        sims = queries @ self.vectors.T
        ids = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(sims, ids, axis=1), ids


def make_embedder(spec: str) -> Embedder:
    """``hashing`` | ``minilm`` | ``mpnet`` | any sentence-transformers model name."""
    presets = {"minilm": MINILM, "mpnet": MPNET}
    if spec == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(presets.get(spec, spec))
