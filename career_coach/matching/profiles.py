"""Resume profiles (from the parser's ontology JSON) and job postings.

Only relationships the parser asserted are used: a Skill node counts for a
candidate only when the Person node lists it in ``hasSkill``; a degree only when
it is linked through ``holdsDegree``. Unknown values stay ``None`` and are never
treated as zero.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .skills import SkillExtractor
from .taxonomy import Taxonomy
from .text import relevant_job_text


ONTOLOGY = "http://example.org/jobmatch#"


@dataclass(frozen=True)
class SkillRef:
    id: str
    label: str


@dataclass
class ResumeProfile:
    cv_id: str
    person_id: str
    skills: list[SkillRef] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)          # normalized O*NET occupation labels
    titles: list[str] = field(default_factory=list)         # original job titles, when available
    years_experience: float | None = None
    degree_rank: int | None = None                          # 1 associate .. 4 doctorate
    evidence: list[str] = field(default_factory=list)       # quoted source spans, when available

    @property
    def skill_ids(self) -> set[str]:
        return {s.id for s in self.skills}

    def text(self) -> str:
        """Retrieval text built only from extracted, validated profile facts."""
        parts = []
        if self.titles or self.roles:
            parts.append("Roles: " + "; ".join(dict.fromkeys(self.titles + self.roles)))
        if self.skills:
            parts.append("Skills: " + ", ".join(s.label for s in self.skills))
        if self.evidence:
            parts.append(" ".join(dict.fromkeys(self.evidence)))
        return "\n".join(parts)


@dataclass
class JobPosting:
    job_id: str
    title: str
    description: str
    company: str = ""
    location: str = ""
    url: str = ""
    required: list[SkillRef] = field(default_factory=list)
    preferred: list[SkillRef] = field(default_factory=list)
    min_years: float | None = None
    min_degree_rank: int | None = None
    source: str = ""

    @property
    def skill_ids(self) -> set[str]:
        return {s.id for s in self.required} | {s.id for s in self.preferred}

    def text(self) -> str:
        return f"{self.title}\n{relevant_job_text(self.description)}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "JobPosting":
        data = dict(data)
        for key in ("required", "preferred"):
            data[key] = [SkillRef(**s) for s in data.get(key, [])]
        return cls(**data)


# ---------------------------------------------------------------------------
# Resumes
# ---------------------------------------------------------------------------

def _local(node_type: str) -> str:
    return node_type.rsplit("#", 1)[-1]


def _label(node: dict, taxonomy: Taxonomy | None) -> str | None:
    metadata = node.get("metadata") or {}
    label = metadata.get("label") or (metadata.get("taxonomy") or {}).get("label")
    if not label and taxonomy is not None:
        label = taxonomy.label(node["id"])   # older outputs carry no metadata
    return label


def local_skill_id(label: str) -> str:
    """ID that ``job_ingestion`` gives a skill with no O*NET match.

    The resume pipeline scopes such IDs to one resume, so they are re-keyed here
    by the normalized label; "CSS" on a resume and in a posting is then one skill.
    """
    key = json.dumps(Taxonomy.match_key(label), sort_keys=True, ensure_ascii=False)
    return "urn:career-coach:local-skill:" + hashlib.sha256(key.encode()).hexdigest()


def _resume(document: dict, taxonomy: Taxonomy | None) -> ResumeProfile:
    nodes = {n["id"]: n for n in document["nodes"]}
    people = [n for n in document["nodes"] if _local(n.get("type", "")) == "Person"]
    if len(people) != 1:
        raise ValueError(f"{document.get('cv_id')}: expected one Person node, found {len(people)}")
    person = people[0]
    profile = ResumeProfile(cv_id=document["cv_id"], person_id=person["id"])

    def evidence(node):
        for item in (node.get("metadata") or {}).get("evidence", []):
            if item.get("text"):
                profile.evidence.append(item["text"])

    for skill_id in person.get("hasSkill", []):
        node = nodes.get(skill_id)
        if node is None:
            raise ValueError(f"{profile.cv_id}: hasSkill references missing node {skill_id}")
        label = _label(node, taxonomy)
        if label:
            identity = local_skill_id(label) if ":local-skill:" in skill_id else skill_id
            if identity not in profile.skill_ids:
                profile.skills.append(SkillRef(identity, label))
            evidence(node)

    years = []
    for experience_id in person.get("hasExperience", []):
        experience = nodes.get(experience_id, {})
        title = (experience.get("metadata") or {}).get("original_title")
        if title and title not in profile.titles:
            profile.titles.append(title)
        years.extend(v for v in experience.get("yearsExperience", []) if isinstance(v, (int, float)))
        for role_id in experience.get("inRole", []):
            label = _label(nodes.get(role_id, {"id": role_id}), taxonomy)
            if label and label not in profile.roles:
                profile.roles.append(label)
    profile.years_experience = float(sum(years)) if years else None

    ranks = []
    for degree_id in person.get("holdsDegree", []):
        for level_id in nodes.get(degree_id, {}).get("degreeLevel", []):
            ranks.extend(nodes.get(level_id, {}).get("rank", []))
    profile.degree_rank = max(ranks) if ranks else None
    return profile


def load_resumes(path, taxonomy: Taxonomy | None = None) -> list[ResumeProfile]:
    """Read the parser's ``{"resumes": [{"cv_id", "nodes"}]}`` output."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = data["resumes"] if isinstance(data, dict) else data
    return [_resume(d, taxonomy) for d in documents]


# ---------------------------------------------------------------------------
# Job postings
# ---------------------------------------------------------------------------

# associate/bachelor/master need degree context ("Scrum Master" is not a degree).
DEGREE_PATTERNS = (
    (1, re.compile(r"\bassociate(?:'s|’s|s)?\s+(?:degree|of|in)\b", re.I)),
    (2, re.compile(r"\b(?:bachelor(?:'s|’s|s)?\s+(?:degree|of|in)\b|b\.?s\.?c?\.?\s+(?:degree\s+)?in\b)", re.I)),
    (3, re.compile(r"\b(?:master(?:'s|’s|s)?\s+(?:degree|of|in)\b|m\.?s\.?c?\.?\s+(?:degree\s+)?in\b)", re.I)),
    (4, re.compile(r"\b(?:ph\.?\s?d\.?|doctorate|doctoral degree)(?![a-z])", re.I)),
)
_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:(?:-|–|to)\s*\d{1,2}\s*)?\+?\s*years?\b(?:\s+of)?(?:\s+[\w/&-]+){0,4}?"
                    r"\s+experience", re.I)


def minimum_years(text: str) -> float | None:
    values = [float(m.group(1)) for m in _YEARS.finditer(text or "")]
    return min(values) if values else None


def minimum_degree(text: str) -> int | None:
    """Lowest degree level a posting mentions (postings list the minimum first)."""
    ranks = [rank for rank, pattern in DEGREE_PATTERNS if pattern.search(text or "")]
    return min(ranks) if ranks else None


def make_job(job_id, title, description, extractor: SkillExtractor, **fields) -> JobPosting:
    extracted = extractor.extract_job_skills(f"{title}\n{description}")
    return JobPosting(
        job_id=str(job_id), title=title or "", description=description or "",
        required=[SkillRef(c.id, c.label) for c in extracted.required],
        preferred=[SkillRef(c.id, c.label) for c in extracted.preferred],
        min_years=minimum_years(description), min_degree_rank=minimum_degree(description), **fields)


def load_kaggle_postings(csv_path, extractor: SkillExtractor, limit: int | None = None) -> list[JobPosting]:
    """Read ``job_postings.csv`` from the Kaggle LinkedIn Job Postings 2023-24 dataset."""
    import pandas as pd

    frame = pd.read_csv(csv_path, nrows=limit, dtype=str, keep_default_na=False)
    missing = {"job_id", "title", "description"} - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    jobs = []
    for row in frame.to_dict(orient="records"):
        if not row["description"].strip():
            continue
        description = row["description"]
        if row.get("skills_desc", "").strip():
            description += "\n\nSkills:\n" + row["skills_desc"]
        jobs.append(make_job(row["job_id"], row["title"], description, extractor,
                             company=row.get("company_name", ""), location=row.get("location", ""),
                             url=row.get("job_posting_url", ""), source="kaggle:arshkon/linkedin-job-postings"))
    return jobs


def save_jobs(jobs: list[JobPosting], path) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        for job in jobs:
            stream.write(json.dumps(job.to_dict(), ensure_ascii=False) + "\n")


def load_jobs(path, extractor: SkillExtractor | None = None) -> list[JobPosting]:
    """Read JSONL postings. Rows without skill lists are extracted when an extractor is given."""
    jobs = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if "required" not in row and "preferred" not in row:
                if extractor is None:
                    raise ValueError(f"Job {row.get('job_id')} has no skills; pass a SkillExtractor")
                extra = {k: row[k] for k in ("company", "location", "url", "source") if k in row}
                jobs.append(make_job(row["job_id"], row["title"], row["description"], extractor, **extra))
            else:
                jobs.append(JobPosting.from_dict(row))
    ids = [j.job_id for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate job_id values")
    return jobs


# ---------------------------------------------------------------------------
# Postings exported by career_coach.job_ingestion (Neo4j-style graph JSON)
# ---------------------------------------------------------------------------

def _values(value):
    return value if isinstance(value, list) else [] if value is None else [value]


def load_graph_jobs(path, postings_csv=None) -> list[JobPosting]:
    """Read ``job_ingestion`` output: one ``{"nodes", "relationships"}`` file per posting.

    ``path`` is a folder of these files or one file. Skills come from the graph's
    ``requiresSkill`` edges (GLiNER + O*NET, as extracted by job_ingestion). The
    export has no description text, so pass the source ``postings.csv`` to give
    BM25 and the dense rankers the full posting; without it they see only titles.
    """
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    jobs = []
    for file in files:
        graph = json.loads(file.read_text(encoding="utf-8"))
        nodes = {n["id"]: n for n in graph.get("nodes", [])}

        def label(node_id):
            return (nodes.get(node_id, {}).get("properties") or {}).get("label") or ""

        for posting in (n for n in graph.get("nodes", []) if "JobPosting" in n.get("labels", [])):
            edges = {}
            for edge in graph.get("relationships", []):
                if edge["start"] == posting["id"]:
                    edges.setdefault(edge["type"], []).append(edge["end"])
            required = [SkillRef(i, label(i)) for i in edges.get("requiresSkill", []) if label(i)]
            ranks = [r for i in edges.get("minDegreeLevel", [])
                     for r in _values((nodes.get(i, {}).get("properties") or {}).get("rank"))]
            years = _values((posting.get("properties") or {}).get("minExperience"))
            jobs.append(JobPosting(
                job_id=posting["id"].rsplit(":", 1)[-1], title=label(posting["id"]), description="",
                company=", ".join(label(i) for i in edges.get("postedBy", [])), required=required,
                min_years=float(min(years)) if years else None, min_degree_rank=min(ranks) if ranks else None,
                source=f"job_ingestion:{file.name}"))
    if postings_csv is not None and jobs:
        _attach_descriptions(jobs, postings_csv)
    ids = [j.job_id for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate job postings")
    return jobs


def _attach_descriptions(jobs: list[JobPosting], csv_path) -> None:
    import pandas as pd

    wanted = {j.job_id for j in jobs}
    by_id = {}
    for chunk in pd.read_csv(csv_path, dtype=str, keep_default_na=False, chunksize=20000,
                             usecols=lambda c: c in {"job_id", "description", "skills_desc", "location",
                                                     "job_posting_url"}):
        for row in chunk[chunk["job_id"].isin(wanted)].to_dict(orient="records"):
            by_id[row["job_id"]] = row
    for job in jobs:
        row = by_id.get(job.job_id)
        if row:
            job.description = row.get("description", "")
            if row.get("skills_desc", "").strip():
                job.description += "\n\nSkills:\n" + row["skills_desc"]
            job.location, job.url = row.get("location", ""), row.get("job_posting_url", "")


def load_any_jobs(path, extractor: SkillExtractor | None = None, postings_csv=None) -> list[JobPosting]:
    """JSONL from the matching ``jobs`` command, or job_ingestion graph JSON (file or folder)."""
    path = Path(path)
    if path.is_dir() or path.suffix == ".json":
        return load_graph_jobs(path, postings_csv)
    return load_jobs(path, extractor)
