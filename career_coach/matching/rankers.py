"""Rankers. Each returns ``[(job_index, score), ...]`` best first, at most ``depth`` long.

Jobs that share no signal with the resume (score <= 0) are left out, so they get
no Reciprocal Rank Fusion credit from that ranker.
"""

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .embeddings import Embedder, VectorIndex
from .profiles import JobPosting, ResumeProfile
from .text import chunk_words, tokenize


Ranked = list[tuple[int, float]]


def _top(scores: dict[int, float], depth: int) -> Ranked:
    items = [(i, s) for i, s in scores.items() if s > 0]
    items.sort(key=lambda pair: (-pair[1], pair[0]))
    return items[:depth]


class _InvertedIndex:
    def __init__(self, documents: list[list[str]]):
        self.size = len(documents)
        self.lengths = [len(d) for d in documents]
        self.average_length = (sum(self.lengths) / self.size) if self.size else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, tokens in enumerate(documents):
            for term, count in Counter(tokens).items():
                self.postings[term].append((index, count))


class TfidfRanker:
    """TF-IDF (sublinear tf, smoothed idf) with cosine similarity."""

    name = "tfidf"

    def fit(self, jobs: list[JobPosting]) -> "TfidfRanker":
        documents = [tokenize(j.text()) for j in jobs]
        self.index = _InvertedIndex(documents)
        n = self.index.size
        self.idf = {t: math.log((1 + n) / (1 + len(p))) + 1 for t, p in self.index.postings.items()}
        norms = [0.0] * n
        self.weights: dict[str, list[tuple[int, float]]] = {}
        for term, postings in self.index.postings.items():
            weighted = [(i, (1 + math.log(c)) * self.idf[term]) for i, c in postings]
            self.weights[term] = weighted
            for i, w in weighted:
                norms[i] += w * w
        self.norms = [math.sqrt(x) or 1.0 for x in norms]
        return self

    def rank(self, resume: ResumeProfile, depth: int = 100) -> Ranked:
        counts = Counter(t for t in tokenize(resume.text()) if t in self.idf)
        query = {t: (1 + math.log(c)) * self.idf[t] for t, c in counts.items()}
        query_norm = math.sqrt(sum(w * w for w in query.values())) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term, qw in query.items():
            for i, w in self.weights[term]:
                scores[i] += qw * w
        return _top({i: s / (query_norm * self.norms[i]) for i, s in scores.items()}, depth)


class BM25Ranker:
    """Okapi BM25 (Robertson & Zaragoza, 2009)."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, jobs: list[JobPosting]) -> "BM25Ranker":
        self.index = _InvertedIndex([tokenize(j.text()) for j in jobs])
        n = self.index.size
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.index.postings.items()}
        return self

    def score_terms(self, terms: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        avg = self.index.average_length or 1.0
        for term in set(terms):
            if term not in self.idf:
                continue
            for i, tf in self.index.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.index.lengths[i] / avg)
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def rank(self, resume: ResumeProfile, depth: int = 100) -> Ranked:
        return _top(self.score_terms(tokenize(resume.text())), depth)


class JaccardRanker:
    """|resume skills ∩ job skills| / |resume skills ∪ job skills| over concept IDs.

    Uses a posting's required skills, or its preferred skills when none are required.
    """

    name = "jaccard"

    def fit(self, jobs: list[JobPosting]) -> "JaccardRanker":
        self.sets = [{s.id for s in (j.required or j.preferred)} for j in jobs]
        self.by_skill: dict[str, list[int]] = defaultdict(list)
        for i, skills in enumerate(self.sets):
            for skill in skills:
                self.by_skill[skill].append(i)
        return self

    def rank(self, resume: ResumeProfile, depth: int = 100) -> Ranked:
        mine = resume.skill_ids
        candidates = {i for s in mine for i in self.by_skill.get(s, [])}
        return _top({i: len(mine & self.sets[i]) / len(mine | self.sets[i]) for i in candidates}, depth)


class DenseRanker:
    """Bi-encoder retrieval: job score = best cosine between any resume chunk and job chunk.

    Postings are chunked (default 200 words, 10% overlap) so long descriptions are
    not truncated by the encoder, and about-us/benefits text is dropped first.
    """

    def __init__(self, embedder: Embedder, name: str | None = None, *, max_words: int = 200,
                 overlap: float = 0.1, use_faiss: bool | None = None, cache_dir=None):
        self.embedder = embedder
        self.name = name or f"dense:{embedder.name}"
        self.max_words, self.overlap, self.use_faiss = max_words, overlap, use_faiss
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _encode_chunks(self) -> np.ndarray:
        """Encode posting chunks, reusing a cached matrix for the same model and chunks."""
        if self.cache_dir is None:
            return self.embedder.encode(self.chunks)
        digest = hashlib.sha256("\x1f".join([self.embedder.name, *self.chunks]).encode()).hexdigest()[:20]
        path = self.cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', self.embedder.name)}-{digest}.npy"
        if path.exists():
            return np.load(path)
        vectors = self.embedder.encode(self.chunks)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def fit(self, jobs: list[JobPosting]) -> "DenseRanker":
        self.chunks, self.owner = [], []
        for i, job in enumerate(jobs):
            for chunk in chunk_words(job.text(), self.max_words, self.overlap) or [job.title]:
                self.chunks.append(chunk)
                self.owner.append(i)
        self.owner = np.asarray(self.owner, dtype=np.int64)
        self.index = VectorIndex(self._encode_chunks(), use_faiss=self.use_faiss)
        return self

    def _query(self, resume: ResumeProfile) -> np.ndarray:
        chunks = chunk_words(resume.text(), self.max_words, self.overlap)
        return self.embedder.encode(chunks) if chunks else np.zeros((0, self.index.vectors.shape[1]), np.float32)

    def chunk_scores(self, resume: ResumeProfile, depth: int) -> dict[int, tuple[float, int]]:
        """job index -> (best similarity, best chunk index)."""
        query = self._query(resume)
        if not len(query):
            return {}
        # Search enough chunks to cover ``depth`` distinct jobs in the common case.
        k = min(len(self.index), max(depth * 4, 50))
        scores, ids = self.index.search(query, k)
        best: dict[int, tuple[float, int]] = {}
        for row_scores, row_ids in zip(scores, ids):
            for score, chunk in zip(row_scores, row_ids):
                if chunk < 0:
                    continue
                job = int(self.owner[chunk])
                if job not in best or score > best[job][0]:
                    best[job] = (float(score), int(chunk))
        return best

    def rank(self, resume: ResumeProfile, depth: int = 100) -> Ranked:
        return _top({j: s for j, (s, _) in self.chunk_scores(resume, depth).items()}, depth)

    def best_chunk(self, resume: ResumeProfile, job_index: int) -> tuple[float, str] | None:
        """Most similar chunk of one posting (for explanations), computed exactly."""
        query = self._query(resume)
        rows = np.flatnonzero(self.owner == job_index)
        if not len(query) or not len(rows):
            return None
        sims = query @ self.index.vectors[rows].T
        flat = int(np.argmax(sims))
        return float(sims.flat[flat]), self.chunks[int(rows[flat % len(rows)])]
