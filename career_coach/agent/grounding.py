"""Grounding context (text-only vs graph) and claim verification for generated text."""

import re
from dataclasses import dataclass, field

from career_coach.matching.matcher import Match, Matcher
from career_coach.matching.profiles import ResumeProfile
from career_coach.matching.skills import SkillExtractor
from career_coach.matching.text import chunk_words


# ---------------------------------------------------------------------------
# Context for retrieval-augmented generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fact:
    id: str      # cited by the model, e.g. "R1", "J2", "G3", "T1"
    text: str


def resume_facts(resume: ResumeProfile) -> list[Fact]:
    facts = []
    if resume.titles or resume.roles:
        facts.append(Fact("R1", "Candidate roles: " + "; ".join(dict.fromkeys(resume.titles + resume.roles))))
    if resume.skills:
        facts.append(Fact(f"R{len(facts) + 1}", "Candidate skills: " + ", ".join(s.label for s in resume.skills)))
    if resume.years_experience is not None:
        facts.append(Fact(f"R{len(facts) + 1}", f"Candidate total years of experience: {resume.years_experience:g}"))
    return facts


def job_facts(match: Match) -> list[Fact]:
    job = match.job
    facts = [Fact("J1", f"Posting {job.job_id}: {job.title}" + (f" at {job.company}" if job.company else ""))]
    if job.required:
        facts.append(Fact("J2", "Required skills: " + ", ".join(s.label for s in job.required)))
    if job.preferred:
        facts.append(Fact(f"J{len(facts) + 1}", "Preferred skills: " + ", ".join(s.label for s in job.preferred)))
    if job.min_years is not None:
        facts.append(Fact(f"J{len(facts) + 1}", f"Posting asks for at least {job.min_years:g} years of experience"))
    return facts


def graph_facts(match: Match) -> list[Fact]:
    graph = match.graph
    if graph is None:
        return []
    lines = graph.paths(match.job)
    if graph.missing_required:
        lines.append("Missing required skills: " + ", ".join(s.label for s in graph.missing_required))
    if graph.missing_preferred:
        lines.append("Missing preferred skills: " + ", ".join(s.label for s in graph.missing_preferred))
    lines.append(f"Knowledge-graph skill score: {graph.score:.2f} (0 to 1)")
    return [Fact(f"G{i}", line) for i, line in enumerate(lines, 1)]


def text_facts(match: Match, resume: ResumeProfile, limit: int = 2) -> list[Fact]:
    """Text-only RAG baseline: raw posting chunks and the resume text, no graph structure."""
    chunks = chunk_words(match.job.text(), 120, 0.1)
    if match.best_chunk and match.best_chunk.get("text"):
        chunks = [match.best_chunk["text"], *chunks]
    facts = [Fact(f"T{i}", c) for i, c in enumerate(list(dict.fromkeys(chunks))[:limit], 1)]
    facts.append(Fact(f"T{len(facts) + 1}", "Resume: " + resume.text()[:1500]))
    return facts


def build_context(match: Match, resume: ResumeProfile, mode: str = "graph") -> list[Fact]:
    """``graph``: structured resume/job facts + graph paths (GraphRAG); ``text``: chunks only."""
    if mode == "graph":
        return resume_facts(resume) + job_facts(match) + graph_facts(match)
    if mode == "text":
        return text_facts(match, resume)
    raise ValueError("mode must be 'graph' or 'text'")


def format_facts(facts: list[Fact]) -> str:
    return "\n".join(f"[{f.id}] {f.text}" for f in facts)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", re.I)


@dataclass
class VerificationResult:
    mentioned_skills: list[str] = field(default_factory=list)
    unsupported_skills: list[str] = field(default_factory=list)
    unsupported_years: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported_skills and not self.unsupported_years

    def to_dict(self) -> dict:
        return {"ok": self.ok, "mentioned_skills": self.mentioned_skills,
                "unsupported_skills": self.unsupported_skills, "unsupported_years": self.unsupported_years}

    def feedback(self) -> str:
        parts = []
        if self.unsupported_skills:
            parts.append("these skills are not supported by the allowed facts: " + ", ".join(self.unsupported_skills))
        if self.unsupported_years:
            parts.append("these experience claims are not supported: " + ", ".join(self.unsupported_years))
        return "; ".join(parts)


class ClaimVerifier:
    """Deterministic checks of O*NET skill mentions and years-of-experience claims.

    Skill mentions are found with the same O*NET phrase rules used for postings, so
    "AWS" and "Amazon Web Services" are the same concept. A mention is supported
    only if its concept ID is in ``allowed``. Skills outside O*NET cannot be checked.
    """

    def __init__(self, extractor: SkillExtractor):
        self.extractor = extractor

    def check(self, text: str, allowed: set[str], *, max_years: float | None = None,
              check_years: bool = False) -> VerificationResult:
        result = VerificationResult()
        for mention in self.extractor.find(text or ""):
            label = mention.concept.label
            if label not in result.mentioned_skills:
                result.mentioned_skills.append(label)
            if mention.concept.id not in allowed and label not in result.unsupported_skills:
                result.unsupported_skills.append(label)
        if check_years:
            for match in _YEARS.finditer(text or ""):
                if max_years is None or float(match.group(1)) > max_years:
                    result.unsupported_years.append(match.group(0))
        return result


def unsupported_skill_rate(results: list[VerificationResult]) -> float:
    """Unsupported skill mentions / all skill mentions, pooled over drafts (0 if none)."""
    mentioned = sum(len(r.mentioned_skills) for r in results)
    return sum(len(r.unsupported_skills) for r in results) / mentioned if mentioned else 0.0


def allowed_for_answer(matcher: Matcher, resume: ResumeProfile, job_ids) -> set[str]:
    """Skills a chat answer may name: the candidate's, plus those of postings it retrieved."""
    allowed = set(resume.skill_ids)
    for job_id in job_ids:
        if job_id in matcher.by_id:
            allowed |= matcher.job(job_id).skill_ids
    return allowed
