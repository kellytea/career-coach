"""Matching systems S1-S7 (the ablation ladder) with explainable results."""

from dataclasses import dataclass, field

from .embeddings import Embedder
from .fusion import reciprocal_rank_fusion
from .graph import GraphEvidence, KGRanker, KGSkillScorer
from .profiles import JobPosting, ResumeProfile
from .rankers import BM25Ranker, DenseRanker, JaccardRanker, Ranked, TfidfRanker


SYSTEMS: dict[str, tuple[str, ...]] = {
    "S1": ("tfidf",),                           # classical
    "S2": ("bm25",),                            # classical
    "S3": ("jaccard",),                         # classical, skills only
    "S4": ("dense:minilm",),                    # deep
    "S5": ("dense:mpnet",),                     # deep
    "S6": ("bm25", "dense:mpnet"),              # hybrid, RRF
    "S7": ("bm25", "dense:mpnet", "kg"),        # hybrid + knowledge graph, RRF (full system)
}


@dataclass
class Match:
    job: JobPosting
    rank: int
    score: float                                     # RRF score, or the single ranker's score
    system: str
    rankers: dict[str, dict] = field(default_factory=dict)   # name -> {"rank", "score"} (None if not retrieved)
    graph: GraphEvidence | None = None
    best_chunk: dict | None = None                   # {"similarity", "text"} from a dense ranker
    requirements: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        graph = self.graph
        return {
            "job_id": self.job.job_id, "title": self.job.title, "company": self.job.company,
            "rank": self.rank, "score": round(self.score, 6), "system": self.system,
            "rankers": self.rankers,
            "kg_score": round(graph.score, 4) if graph else None,
            "matched_skills": [s.label for s in (graph.matched_required + graph.matched_preferred)] if graph else [],
            "near_miss_skills": [{"job_skill": n.job_skill.label, "your_skill": n.resume_skill.label,
                                  "similarity": round(n.similarity, 3)} for n in graph.near_misses] if graph else [],
            "missing_required": [s.label for s in graph.missing_required] if graph else [],
            "missing_preferred": [s.label for s in graph.missing_preferred] if graph else [],
            "best_matching_text": self.best_chunk,
            "requirements": self.requirements,
            "url": self.job.url,
        }


def _requirement(required, actual):
    if required is None:
        return {"required": None, "yours": actual, "status": "not stated"}
    if actual is None:
        return {"required": required, "yours": None, "status": "unknown from resume"}
    return {"required": required, "yours": actual, "status": "meets" if actual >= required else "below"}


class Matcher:
    def __init__(self, jobs: list[JobPosting], rankers: dict, kg_scorer: KGSkillScorer | None = None, *,
                 depth: int = 100, rrf_k: int = 60):
        if not jobs:
            raise ValueError("Matcher needs at least one job posting")
        self.jobs = jobs
        self.by_id = {j.job_id: i for i, j in enumerate(jobs)}
        self.depth, self.rrf_k = depth, rrf_k
        self.kg_scorer = kg_scorer or KGSkillScorer()
        self.rankers = {name: ranker.fit(jobs) for name, ranker in rankers.items()}

    @classmethod
    def build(cls, jobs: list[JobPosting], *, minilm: Embedder | None = None, mpnet: Embedder | None = None,
              kg_embedder: Embedder | None = None, tau: float = 0.7, depth: int = 100, rrf_k: int = 60,
              use_faiss: bool | None = None, cache_dir=None) -> "Matcher":
        """Create every ranker whose model is available. Without ``mpnet``, S5-S7 are unavailable."""
        scorer = KGSkillScorer(kg_embedder or mpnet or minilm, tau=tau)
        rankers = {"tfidf": TfidfRanker(), "bm25": BM25Ranker(), "jaccard": JaccardRanker(), "kg": KGRanker(scorer)}
        if minilm is not None:
            rankers["dense:minilm"] = DenseRanker(minilm, "dense:minilm", use_faiss=use_faiss, cache_dir=cache_dir)
        if mpnet is not None:
            rankers["dense:mpnet"] = DenseRanker(mpnet, "dense:mpnet", use_faiss=use_faiss, cache_dir=cache_dir)
        return cls(jobs, rankers, scorer, depth=depth, rrf_k=rrf_k)

    def available_systems(self) -> list[str]:
        return [s for s, parts in SYSTEMS.items() if all(p in self.rankers for p in parts)]

    def job(self, job_id: str) -> JobPosting:
        if job_id not in self.by_id:
            raise KeyError(f"Unknown job_id {job_id}")
        return self.jobs[self.by_id[job_id]]

    def _parts(self, system: str) -> tuple[str, ...]:
        if system not in SYSTEMS:
            raise KeyError(f"Unknown system {system}; choose one of {sorted(SYSTEMS)}")
        missing = [p for p in SYSTEMS[system] if p not in self.rankers]
        if missing:
            raise ValueError(f"{system} needs rankers {missing}; build the Matcher with those embedders")
        return SYSTEMS[system]

    def _lists(self, resume: ResumeProfile, parts) -> dict[str, Ranked]:
        return {name: self.rankers[name].rank(resume, self.depth) for name in parts}

    def _order(self, lists: dict[str, Ranked]) -> list[tuple[int, float]]:
        if len(lists) == 1:
            return list(next(iter(lists.values())))
        fused = reciprocal_rank_fusion({n: [i for i, _ in l] for n, l in lists.items()}, k=self.rrf_k)
        return [(i, score) for i, score, _ in fused]

    def _match(self, resume, index, rank, score, system, lists) -> Match:
        job = self.jobs[index]
        rankers = {}
        for name, ranked in lists.items():
            position = next((p for p, (i, _) in enumerate(ranked, 1) if i == index), None)
            value = next((s for i, s in ranked if i == index), None)
            rankers[name] = {"rank": position, "score": None if value is None else round(value, 6)}
        dense = next((self.rankers[n] for n in (*lists, "dense:mpnet", "dense:minilm")
                      if n.startswith("dense:") and n in self.rankers), None)
        chunk = dense.best_chunk(resume, index) if dense is not None else None
        return Match(
            job=job, rank=rank, score=score, system=system, rankers=rankers,
            graph=self.kg_scorer.evidence(resume, job),
            best_chunk={"similarity": round(chunk[0], 4), "text": chunk[1][:600]} if chunk else None,
            requirements={"years_experience": _requirement(job.min_years, resume.years_experience),
                          "degree_level": _requirement(job.min_degree_rank, resume.degree_rank)})

    def rank(self, resume: ResumeProfile, system: str = "S7", top_k: int = 10, where=None) -> list[Match]:
        """Top matches; ``where(job) -> bool`` filters the ranked list before cutting to ``top_k``."""
        lists = self._lists(resume, self._parts(system))
        order = [(i, s) for i, s in self._order(lists) if where is None or where(self.jobs[i])][:top_k]
        return [self._match(resume, i, r, s, system, lists) for r, (i, s) in enumerate(order, 1)]

    def ranked_ids(self, resume: ResumeProfile, system: str, top_k: int) -> list[str]:
        """Job IDs only, for evaluation (no explanations computed)."""
        order = self._order(self._lists(resume, self._parts(system)))[:top_k]
        return [self.jobs[i].job_id for i, _ in order]

    def explain(self, resume: ResumeProfile, job_id: str, system: str = "S7") -> Match:
        """Explanation for one posting, including where each ranker placed it."""
        index = self.by_id.get(job_id)
        if index is None:
            raise KeyError(f"Unknown job_id {job_id}")
        lists = self._lists(resume, self._parts(system))
        order = self._order(lists)
        position = next((p for p, (i, _) in enumerate(order, 1) if i == index), None)
        score = next((s for i, s in order if i == index), 0.0)
        return self._match(resume, index, position, score, system, lists)
