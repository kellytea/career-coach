"""Knowledge-graph skill score (the third ranker fused with RRF).

    g(c, j) = (w_r * sum_{r in R_j} m(r, c) + w_p * sum_{p in P_j} m(p, c)) / (w_r |R_j| + w_p |P_j|)

R_j / P_j are a posting's required / preferred skills, w_r = 2 and w_p = 1 by default.
m(k, c) = 1 when the candidate has skill k (same concept ID, i.e. Person-hasSkill->k
and JobPosting-requiresSkill->k share a node), ``partial_credit`` (0.5) when the
candidate has a skill whose Sentence-BERT cosine similarity with k is >= tau, else 0.
g is in [0, 1]; postings with no extracted skills get no score.

tau should be tuned on a development split of the labeled benchmark, never on the
test resumes. The default 0.7 is a starting point, not a validated value.
"""

from dataclasses import dataclass, field

import numpy as np

from .embeddings import Embedder
from .profiles import JobPosting, ResumeProfile, SkillRef
from .rankers import Ranked, _top


@dataclass(frozen=True)
class NearMiss:
    job_skill: SkillRef
    resume_skill: SkillRef
    similarity: float


@dataclass
class GraphEvidence:
    score: float
    matched_required: list[SkillRef] = field(default_factory=list)
    matched_preferred: list[SkillRef] = field(default_factory=list)
    near_misses: list[NearMiss] = field(default_factory=list)
    missing_required: list[SkillRef] = field(default_factory=list)
    missing_preferred: list[SkillRef] = field(default_factory=list)

    def paths(self, job: JobPosting) -> list[str]:
        """Human-readable graph paths that support the score."""
        lines = [f"Candidate -hasSkill-> {s.label} <-requiresSkill- {job.title} [{job.job_id}]"
                 for s in self.matched_required]
        lines += [f"Candidate -hasSkill-> {s.label} <-prefersSkill- {job.title} [{job.job_id}]"
                  for s in self.matched_preferred]
        lines += [f"Candidate -hasSkill-> {n.resume_skill.label} ~similar({n.similarity:.2f})~ "
                  f"{n.job_skill.label} <-(required/preferred)- {job.title} [{job.job_id}]" for n in self.near_misses]
        return lines


class KGSkillScorer:
    def __init__(self, embedder: Embedder | None = None, *, tau: float = 0.7, required_weight: float = 2.0,
                 preferred_weight: float = 1.0, partial_credit: float = 0.5):
        if not 0 <= partial_credit <= 1 or required_weight <= 0 or preferred_weight < 0:
            raise ValueError("Invalid KG score weights")
        self.embedder, self.tau = embedder, tau
        self.required_weight, self.preferred_weight, self.partial_credit = (
            required_weight, preferred_weight, partial_credit)
        self._vectors: dict[str, np.ndarray] = {}

    def _vector(self, labels: list[str]) -> np.ndarray:
        missing = [l for l in dict.fromkeys(labels) if l not in self._vectors]
        if missing:
            for label, vector in zip(missing, self.embedder.encode(missing)):
                self._vectors[label] = vector
        return np.stack([self._vectors[l] for l in labels])

    def warm(self, labels) -> None:
        """Pre-compute label embeddings (e.g. every job skill) in one batch."""
        if self.embedder is not None:
            labels = list(dict.fromkeys(labels))
            if labels:
                self._vector(labels)

    def near_map(self, resume: ResumeProfile, skills: list[SkillRef]) -> dict[str, NearMiss]:
        """Skills (not held) whose best similarity to a held skill is >= tau."""
        mine = resume.skill_ids
        todo = list({s.id: s for s in skills if s.id not in mine}.values())
        if not todo or not resume.skills or self.embedder is None:
            return {}
        sims = self._vector([s.label for s in todo]) @ self._vector([s.label for s in resume.skills]).T
        near = {}
        for skill, row in zip(todo, sims):
            best = int(np.argmax(row))
            if row[best] >= self.tau:
                near[skill.id] = NearMiss(skill, resume.skills[best], float(row[best]))
        return near

    def evidence(self, resume: ResumeProfile, job: JobPosting,
                 near: dict[str, NearMiss] | None = None) -> GraphEvidence:
        if near is None:
            near = self.near_map(resume, [*job.required, *job.preferred])
        mine = resume.skill_ids
        result = GraphEvidence(score=0.0)
        for skills, matched, missing in ((job.required, result.matched_required, result.missing_required),
                                         (job.preferred, result.matched_preferred, result.missing_preferred)):
            for skill in skills:
                if skill.id in mine:
                    matched.append(skill)
                elif skill.id in near:
                    result.near_misses.append(near[skill.id])
                else:
                    missing.append(skill)

        def credit(skills):
            return sum(1.0 if s.id in mine else self.partial_credit if s.id in near else 0.0 for s in skills)

        denominator = self.required_weight * len(job.required) + self.preferred_weight * len(job.preferred)
        if denominator:
            result.score = (self.required_weight * credit(job.required)
                            + self.preferred_weight * credit(job.preferred)) / denominator
        return result


class KGRanker:
    name = "kg"

    def __init__(self, scorer: KGSkillScorer):
        self.scorer = scorer

    def fit(self, jobs: list[JobPosting]) -> "KGRanker":
        self.jobs = jobs
        self.by_skill: dict[str, list[int]] = {}
        self.skills: dict[str, SkillRef] = {}
        for i, job in enumerate(jobs):
            for skill in (*job.required, *job.preferred):
                self.by_skill.setdefault(skill.id, []).append(i)
                self.skills.setdefault(skill.id, skill)
        self.scorer.warm([s.label for s in self.skills.values()])
        return self

    def rank(self, resume: ResumeProfile, depth: int = 100) -> Ranked:
        # Similarities are computed once per resume over all distinct job skills.
        near = self.scorer.near_map(resume, list(self.skills.values()))
        candidates = {i for s in (*resume.skill_ids, *near) for i in self.by_skill.get(s, [])}
        return _top({i: self.scorer.evidence(resume, self.jobs[i], near).score for i in candidates}, depth)
