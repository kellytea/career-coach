"""Shared, offline fixtures for the matching and agent tests (no model downloads, no network)."""

import importlib.util
import json
import sys
import types
from functools import lru_cache
from pathlib import Path

# The matching code reuses the parser's TaxonomyResolver. Importing that package
# also imports PyMuPDF, which these tests never use; stub it only if it is absent.
if importlib.util.find_spec("pymupdf") is None:
    sys.modules.setdefault("pymupdf", types.ModuleType("pymupdf"))

import numpy as np

from career_coach.matching.taxonomy import Taxonomy, concept_id
from career_coach.matching.skills import SkillExtractor


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_JOBS = FIXTURES / "sample_jobs.jsonl"
LEGACY_OUTPUT = FIXTURES / "legacy_resume_output.json"   # parser output from before metadata was added
ONTOLOGY = "http://example.org/jobmatch#"


@lru_cache(maxsize=1)
def taxonomy() -> Taxonomy:
    return Taxonomy()


@lru_cache(maxsize=1)
def extractor() -> SkillExtractor:
    return SkillExtractor(taxonomy())


def concept(label: str, kind: str = "skill"):
    matches = [c for c in taxonomy().by_id.values() if c.label == label and c.kind == kind]
    if len(matches) != 1:
        raise LookupError(f"{label!r} is not a unique {kind} concept")
    return matches[0]


def _node(kind, identity, **fields):
    return {"type": ONTOLOGY + kind, "id": identity, **fields}


def resume_document(cv_id="candidate.pdf", skills=("Python", "Docker", "Oracle Java", "React"),
                    unlinked_skills=("Kubernetes",), role="Software Developers", title="Software Engineer",
                    years=(2.0, 1.5), degree_rank=2) -> dict:
    """A parser-format resume with metadata, as the current pipeline writes it."""
    base = f"urn:career-coach:resume:test-{cv_id}"
    nodes, skill_ids = [], []
    for label in (*skills, *unlinked_skills):
        c = concept(label)
        nodes.append(_node("Skill", c.id, metadata={
            "label": c.label, "taxonomy": {"id": c.taxonomy_id, "namespace": c.namespace, "label": c.label,
                                           "source_version": c.version},
            "evidence": [{"entity_id": "e", "source_document": cv_id, "source_field": "section_texts.skills",
                          "start": 0, "end": len(label), "text": label, "label": "skill"}]}))
        if label in skills:
            skill_ids.append(c.id)
    role_concept = concept(role, "occupation")
    nodes.append(_node("Role", role_concept.id, metadata={"label": role_concept.label}))
    experiences = []
    for i, value in enumerate(years):
        identity = f"{base}:experience:{i}"
        experiences.append(identity)
        nodes.append(_node("Experience", identity, atOrganization=[], inRole=[role_concept.id],
                           yearsExperience=[value], metadata={"original_title": title}))
    degrees = []
    if degree_rank is not None:
        level = f"urn:career-coach:degree-level:v1:rank{degree_rank}"
        nodes.append(_node("DegreeLevel", level, rank=[degree_rank]))
        degrees.append(f"{base}:degree:0")
        nodes.append(_node("Degree", degrees[0], degreeLevel=[level]))
    nodes.append(_node("Person", f"{base}:person", hasSkill=skill_ids, hasExperience=experiences,
                       holdsDegree=degrees))
    return {"cv_id": cv_id, "nodes": nodes}


def write_resumes(path: Path, *documents) -> Path:
    path.write_text(json.dumps({"resumes": list(documents)}), encoding="utf-8")
    return path


class TableEmbedder:
    """Embedder with hand-set vectors, so similarity values in tests are exact.

    Texts not in the table get a zero vector (cosine 0 with everything).
    """

    def __init__(self, table: dict[str, list[float]], dim: int = 4, name: str = "table"):
        self.table, self.dim, self.name = table, dim, name

    def encode(self, texts):
        rows = []
        for text in texts:
            vector = np.asarray(self.table.get(text, [0.0] * self.dim), dtype=np.float32)
            rows.append(vector / (np.linalg.norm(vector) or 1))
        return np.stack(rows) if rows else np.zeros((0, self.dim), np.float32)


__all__ = ["concept", "concept_id", "extractor", "resume_document", "taxonomy", "write_resumes",
           "TableEmbedder", "SAMPLE_JOBS", "LEGACY_OUTPUT"]
