"""Stage 7: Pydantic contracts guided by data/ontologies/resume-job.ttl.

These are application validation models, not an OWL reasoner or SHACL engine.
Evidence, taxonomy and reference checks run locally without inference or repair.
"""

import hashlib
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from . import models as dto
from .taxonomy import Taxonomy, match_key


VALIDATOR_VERSION = "0.1.0"
NAMESPACE = "http://example.org/jobmatch#"
Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
Years = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ValidationModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True, allow_inf_nan=False)


class EvidenceSpan(ValidationModel):
    mention_id: Identifier
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Evidence end must follow its start")
        return self


class OntologyEntity(ValidationModel):
    id: Identifier
    label: str | None = None
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    ontology_class: ClassVar[str]


class Organization(OntologyEntity):
    ontology_class = NAMESPACE + "Organization"


class Role(OntologyEntity):
    ontology_class = NAMESPACE + "Role"


class Skill(OntologyEntity):
    ontology_class = NAMESPACE + "Skill"


class DegreeLevel(OntologyEntity):
    ontology_class = NAMESPACE + "DegreeLevel"
    # The ontology specifies an integer, but does not define a degree ranking system.
    rank: int | None = None


class Degree(OntologyEntity):
    ontology_class = NAMESPACE + "Degree"
    degree_level: list[Identifier] = Field(default_factory=list, alias="degreeLevel")


class Experience(OntologyEntity):
    ontology_class = NAMESPACE + "Experience"
    at_organization: list[Identifier] = Field(default_factory=list, alias="atOrganization")
    in_role: list[Identifier] = Field(default_factory=list, alias="inRole")
    # Non-negative, finite years is an application rule in addition to xsd:float.
    years_experience: Years | None = Field(default=None, alias="yearsExperience")


class Person(OntologyEntity):
    ontology_class = NAMESPACE + "Person"
    has_experience: list[Identifier] = Field(default_factory=list, alias="hasExperience")
    has_skill: list[Identifier] = Field(default_factory=list, alias="hasSkill")
    holds_degree: list[Identifier] = Field(default_factory=list, alias="holdsDegree")


class JobPosting(OntologyEntity):
    ontology_class = NAMESPACE + "JobPosting"
    for_role: list[Identifier] = Field(default_factory=list, alias="forRole")
    posted_by: list[Identifier] = Field(default_factory=list, alias="postedBy")
    requires_skill: list[Identifier] = Field(default_factory=list, alias="requiresSkill")
    min_degree_level: list[Identifier] = Field(default_factory=list, alias="minDegreeLevel")
    min_experience: Years | None = Field(default=None, alias="minExperience")


class OntologyDocument(ValidationModel):
    """Flat, referential ontology view of supported positive extracted assertions."""

    persons: list[Person] = Field(default_factory=list)
    organizations: list[Organization] = Field(default_factory=list)
    roles: list[Role] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    degree_levels: list[DegreeLevel] = Field(default_factory=list)
    degrees: list[Degree] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    job_postings: list[JobPosting] = Field(default_factory=list)

    @model_validator(mode="after")
    def references_match_ontology_ranges(self):
        groups = {name: {node.id for node in getattr(self, name)} for name in type(self).model_fields}
        all_ids = [node.id for name in groups for node in getattr(self, name)]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Ontology entity IDs must be unique across all classes")
        ranges = {
            "persons": {"has_experience": "experiences", "has_skill": "skills", "holds_degree": "degrees"},
            "experiences": {"at_organization": "organizations", "in_role": "roles"},
            "degrees": {"degree_level": "degree_levels"},
            "job_postings": {"for_role": "roles", "posted_by": "organizations", "requires_skill": "skills",
                             "min_degree_level": "degree_levels"},
        }
        for collection, fields in ranges.items():
            for node in getattr(self, collection):
                for name, target in fields.items():
                    references = getattr(node, name)
                    if len(references) != len(set(references)):
                        raise ValueError("An ontology property contains duplicate references")
                    if not set(references).issubset(groups[target]):
                        raise ValueError("An ontology property references a missing entity or the wrong class")
        return self


class Finding(ValidationModel):
    severity: Literal["error", "warning"]
    code: str
    path: str
    message: str


class ValidationReport(ValidationModel):
    validator_version: str = VALIDATOR_VERSION
    status: Literal["valid", "needs_review", "invalid", "error"] = "valid"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_sha256: str | None = None
    ontology_sha256: str | None = None
    model_schema_sha256: str | None = None
    taxonomy_version: str | None = None
    taxonomy_manifest_sha256: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    ontology_data: OntologyDocument | None = None


@lru_cache(maxsize=2)
def _taxonomy(directory, manifest_sha256):
    # Validate against the pinned lookups, with no embedding or model loading.
    return Taxonomy(directory)


def _ontology_view(extraction, kind):
    """Map supported assertions only; leave strength, polarity and alternatives in raw records."""
    graph = OntologyDocument()
    entity_for = {}
    classes = {"organization": (Organization, graph.organizations), "job title": (Role, graph.roles),
               "skill": (Skill, graph.skills), "degree": (Degree, graph.degrees)}
    for mention in extraction.mentions:
        if mention.label not in classes:
            continue
        cls, collection = classes[mention.label]
        node = cls(id=f"{cls.__name__}:{mention.id}", label=mention.text,
                   evidence=[EvidenceSpan(mention_id=mention.id, start=mention.start, end=mention.end)])
        collection.append(node)
        entity_for[mention.id] = node.id
    if kind == "resume":
        root = Person(id=f"Person:{extraction.document_version_id}")
        graph.persons.append(root)
        for claim in extraction.claims:
            if claim.polarity != "positive" or claim.mention_id not in entity_for:
                continue
            if claim.kind == "skill":
                root.has_skill.append(entity_for[claim.mention_id])
            elif claim.kind == "degree":
                root.holds_degree.append(entity_for[claim.mention_id])
        mentions = {m.id: m for m in extraction.mentions}
        entries = {e.id: e for e in extraction.entries}
        for relation in extraction.relations:
            if relation.predicate != "hasExperience" or entries[relation.object].ambiguous:
                continue
            anchors = [link for link in extraction.relations if link.subject == relation.object
                       and link.predicate in {"inRole", "atOrganization"}]
            if any(mentions[link.object].polarity != "positive" for link in anchors):
                continue
            experience = Experience(id=f"Experience:{relation.object}")
            for link in anchors:
                mention = mentions[link.object]
                if mention.polarity != "positive":
                    continue
                target = experience.in_role if link.predicate == "inRole" else experience.at_organization
                target.append(entity_for[mention.id])
                experience.evidence.append(EvidenceSpan(mention_id=mention.id, start=mention.start, end=mention.end))
            graph.experiences.append(experience)
            root.has_experience.append(experience.id)
    else:
        root = JobPosting(id=f"JobPosting:{extraction.document_version_id}")
        graph.job_postings.append(root)
        for relation in extraction.relations:
            if (relation.predicate in {"forRole", "postedBy"}
                    and next(m for m in extraction.mentions if m.id == relation.object).polarity == "positive"):
                target = root.for_role if relation.predicate == "forRole" else root.posted_by
                target.append(entity_for[relation.object])
        for requirement in extraction.requirements:
            if (requirement.kind == "skill" and requirement.strength == "required"
                    and requirement.polarity == "positive" and not requirement.alternatives_present):
                root.requires_skill.append(entity_for[requirement.mention_id])
    # Revalidate after assembly: mutating a list is not assignment validation in Pydantic.
    return OntologyDocument.model_validate(graph.model_dump(by_alias=True), strict=True)


def validate_extraction(payload, *, kind, version_id, ontology_path, taxonomy_dir):
    """Return a persisted-report-ready value; never mutate the extraction or call an LLM."""
    report = ValidationReport()

    def issue(code, path, message, severity="error"):
        report.findings.append(Finding(severity=severity, code=code, path=path, message=message))

    def finish():
        report.status = ("invalid" if any(f.severity == "error" for f in report.findings)
                         else "needs_review" if report.findings else "valid")
        return report

    try:
        report.extraction_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                                            allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError):
        issue("schema_error", "$", "Extraction must contain finite, JSON-serializable values")
        return finish()
    try:
        report.ontology_sha256 = hashlib.sha256(Path(ontology_path).read_bytes()).hexdigest()
        schema = json.dumps(OntologyDocument.model_json_schema(), sort_keys=True).encode()
        report.model_schema_sha256 = hashlib.sha256(schema).hexdigest()
        report.taxonomy_manifest_sha256 = hashlib.sha256((Path(taxonomy_dir) / "manifest.json").read_bytes()).hexdigest()
        taxonomy = _taxonomy(str(Path(taxonomy_dir).resolve()), report.taxonomy_manifest_sha256)
        report.taxonomy_version = taxonomy.version
    except (OSError, ValueError, TypeError, KeyError):
        report.status = "error"
        issue("validation_context_unavailable", "$", "Cannot load validation inputs or pinned reference data")
        return report
    try:
        extraction = dto.Extraction.model_validate(payload, strict=True)
    except ValidationError as error:
        for detail in error.errors(include_input=False, include_context=False, include_url=False):
            issue("schema_error", ".".join(map(str, detail["loc"])), detail["msg"])
        return finish()
    if kind not in {"resume", "job"} or extraction.document_version_id != version_id:
        issue("document_identity", "document_version_id", "Document kind or version identity is inconsistent")
    if hashlib.sha256(extraction.text.encode()).hexdigest() != extraction.text_sha256:
        issue("text_checksum", "text_sha256", "Saved text does not match its recorded checksum")
    if not extraction.text.strip():
        issue("empty_text", "text", "Saved extraction text is empty")
    if not extraction.mentions:
        issue("no_mentions", "mentions", "No entity mentions were extracted; review coverage", "warning")
    if extraction.provenance.get("ontology_sha256") != report.ontology_sha256:
        issue("ontology_changed", "provenance.ontology_sha256", "Ontology differs from the extraction's recorded version", "warning")
    if extraction.provenance.get("taxonomy_version") != taxonomy.version:
        issue("taxonomy_version", "provenance.taxonomy_version", "Taxonomy release does not match validation reference data")
    if extraction.provenance.get("taxonomy_manifest_sha256") != report.taxonomy_manifest_sha256:
        issue("taxonomy_manifest_changed", "provenance.taxonomy_manifest_sha256", "Taxonomy manifest differs from the extraction's recorded inputs", "warning")

    # DTO defaults support extraction construction. At this boundary, IDs and fields
    # must already be present; never accept regenerated IDs or silently dropped fields.
    groups = {"blocks": dto.Block, "mentions": dto.Mention, "entries": dto.Entry,
              "claims": dto.Claim, "requirements": dto.Requirement, "relations": dto.Relation,
              "resolutions": dto.Resolution}
    unknown = set(payload) - set(dto.Extraction.model_fields)
    if unknown:
        issue("unknown_fields", "$", "Extraction has unsupported top-level fields")
    if set(dto.Extraction.model_fields) - set(payload):
        issue("missing_fields", "$", "Saved extraction is missing serialized contract fields")
    all_ids = set()
    for collection, cls in groups.items():
        for index, record in enumerate(payload.get(collection, [])):
            path = f"{collection}.{index}"
            if set(record) - set(cls.model_fields):
                issue("unknown_fields", path, "Record has unsupported fields")
            if set(cls.model_fields) - set(record):
                issue("missing_fields", path, "Saved record is missing serialized contract fields")
            if collection == "resolutions":
                for candidate_index, candidate in enumerate(record.get("candidates", [])):
                    if set(candidate) - set(dto.Candidate.model_fields):
                        issue("unknown_fields", f"{path}.candidates.{candidate_index}", "Candidate has unsupported fields")
                    if set(dto.Candidate.model_fields) - set(candidate):
                        issue("missing_fields", f"{path}.candidates.{candidate_index}", "Saved candidate is missing serialized contract fields")
            if collection != "resolutions":
                record_id = record.get("id")
                if not isinstance(record_id, str) or not record_id.strip() or record_id in all_ids:
                    issue("record_identity", path + ".id", "Record ID is missing, blank or duplicated")
                else:
                    all_ids.add(record_id)
    if any(f.severity == "error" for f in report.findings):
        return finish()
    blocks = {b.id: b for b in extraction.blocks}
    mentions = {m.id: m for m in extraction.mentions}
    entries = {e.id: e for e in extraction.entries}
    claims = {c.id: c for c in extraction.claims}
    requirements = {r.id: r for r in extraction.requirements}

    def span(start, end, path):
        if not 0 <= start < end <= len(extraction.text):
            issue("evidence_bounds", path, "Span is outside the saved text or has an invalid length")
            return False
        return True

    def covers(start, end, target):
        return start <= target.start and target.end <= end

    def score(value, method, path):
        low = -1 if method == "embedding_cosine" else 0
        if not math.isfinite(value) or not low <= value <= 1:
            issue("score_bounds", path, "Score is non-finite or outside the range for its method")

    for block in extraction.blocks:
        span(block.start, block.end, f"blocks.{block.id}")
        if block.page is not None and block.page < 1:
            issue("page_number", f"blocks.{block.id}.page", "Page numbers must be positive")
    ontology_types = {"job title": Role, "skill": Skill, "organization": Organization, "degree": Degree}
    for mention in extraction.mentions:
        path = f"mentions.{mention.id}"
        span(mention.start, mention.end, path)
        if extraction.text[mention.start:mention.end] != mention.text:
            issue("evidence_text", path, "Mention text does not match its saved text span")
        covering = [blocks[bid] for bid in mention.block_ids if bid in blocks]
        if (not covering or len(covering) != len(mention.block_ids) or mention.block_id != mention.block_ids[0]
                or len(set(mention.block_ids)) != len(mention.block_ids)):
            issue("block_reference", path, "Mention has missing, duplicate or inconsistent block references")
        elif (not covers(covering[0].start, covering[-1].end, mention)
              or any(b.heading for b in covering) or len({(b.section, b.page) for b in covering}) != 1):
            issue("mention_context", path, "Mention crosses incompatible blocks or is not covered by them")
        expected = ontology_types.get(mention.label)
        if mention.label not in {*ontology_types, "date", "date range", "experience duration", "location", "certification"}:
            issue("mention_label", path + ".label", "Mention label is not supported by the extraction contract")
        if mention.ontology_type != (expected.ontology_class if expected else None):
            issue("ontology_type", path + ".ontology_type", "Mention's ontology class is incompatible with its label")
        if mention.score is not None:
            score(mention.score, "entity_confidence", path + ".score")
    for entry in extraction.entries:
        path = f"entries.{entry.id}"
        span(entry.start, entry.end, path)
        grouped = [blocks[bid] for bid in entry.block_ids if bid in blocks]
        if (not grouped or len(grouped) != len(entry.block_ids) or len(set(entry.block_ids)) != len(entry.block_ids)
                or entry.start != grouped[0].start or entry.end != grouped[-1].end
                or any(b.heading for b in grouped) or len({(b.section, b.page) for b in grouped}) != 1):
            issue("entry_reference", path, "Entry's block references or evidence boundaries are inconsistent")
        if entry.ambiguous:
            issue("ambiguous_entry", path, "Entry associations remain ambiguous", "warning")
    for collection in ("claims", "requirements"):
        for record in getattr(extraction, collection):
            path = f"{collection}.{record.id}"
            span(record.evidence_start, record.evidence_end, path)
            mention = mentions.get(record.mention_id)
            expected = "experience duration" if record.kind == "experience" else record.kind
            if mention is None or mention.label != expected:
                issue("mention_reference", path, "Assertion references a missing mention or incompatible entity kind")
            elif not covers(record.evidence_start, record.evidence_end, mention) or record.polarity != mention.polarity:
                issue("assertion_evidence", path, "Assertion evidence or polarity is inconsistent with its mention")
            if collection == "claims" and record.entry_id is not None:
                entry = entries.get(record.entry_id)
                if entry is None or entry.ambiguous or (mention and not covers(entry.start, entry.end, mention)):
                    issue("claim_entry", path, "Claim refers to an absent, ambiguous or incompatible entry")
            if collection == "requirements" and record.duration_mention_id is not None:
                duration = mentions.get(record.duration_mention_id)
                if duration is None or duration.label != "experience duration" or not covers(record.evidence_start, record.evidence_end, duration):
                    issue("requirement_duration", path, "Requirement duration is missing, incorrectly typed or outside its evidence")
            if record.kind == "certification":
                issue("ontology_gap", path, "Base ontology has no Certification class; original assertion retained", "warning")
            if collection == "requirements" and (record.strength != "required" or record.alternatives_present or record.kind == "experience"):
                issue("requirement_context", path, "Requirement context remains in the source record rather than an unconditional ontology property", "warning")
    if (kind == "resume" and extraction.requirements) or (kind == "job" and (extraction.claims or extraction.entries)):
        issue("document_scope", "$", "Resume claims/entries and job requirements use incompatible document scopes")

    _check_relations(extraction, kind, mentions, entries, claims, requirements, span, covers, issue)
    _check_resolutions(extraction, mentions, taxonomy, score, issue)
    if not any(f.severity == "error" for f in report.findings):
        try:
            report.ontology_data = _ontology_view(extraction, kind)
        except ValidationError as error:
            for detail in error.errors(include_input=False, include_context=False, include_url=False):
                issue("ontology_model", ".".join(map(str, detail["loc"])), detail["msg"])
    return finish()


def _check_relations(extraction, kind, mentions, entries, claims, requirements, span, covers, issue):
    root = "document_subject" if kind == "resume" else "job_posting"
    root_type = "Person" if kind == "resume" else "JobPosting"
    mention_types = {"job title": "Role", "organization": "Organization", "skill": "Skill", "degree": "Degree",
                     "date range": "DateRange", "date": "Date", "experience duration": "Duration",
                     "certification": "Certification", "location": "Location"}
    types = {root: root_type}
    types.update({m.id: mention_types.get(m.label, "Unknown") for m in mentions.values()})
    types.update({e.id: "Experience" if e.kind == "experience" else "Entry" for e in entries.values()})
    types.update({c.id: "SkillClaim" if c.kind == "skill" else "CredentialClaim" for c in claims.values()})
    types.update({r.id: "Requirement" for r in requirements.values()})
    roots, claim_types = {"Person", "JobPosting"}, {"SkillClaim", "CredentialClaim"}
    rules = {
        "hasSkillClaim": ({"Person"}, {"SkillClaim"}),
        "hasCredentialClaim": ({"Person"}, {"CredentialClaim"}),
        "hasRequirement": ({"JobPosting"}, {"Requirement"}),
        "aboutMention": (claim_types | {"Requirement"}, set(mention_types.values())),
        "inEntry": (claim_types, {"Experience", "Entry"}),
        "hasExperience": ({"Person"}, {"Experience"}),
        "inRole": ({"Experience"}, {"Role"}),
        "atOrganization": ({"Experience"}, {"Organization"}),
        "hasDateRange": ({"Experience"}, {"DateRange"}),
        "forRole": ({"JobPosting"}, {"Role"}),
        "postedBy": ({"JobPosting"}, {"Organization"}),
        "mentionsRole": (roots, {"Role"}),
        "mentionsOrganization": (roots, {"Organization"}),
        "mentionsLocation": (roots, {"Location"}),
        "mentionsSkill": (roots, {"Skill"}),
        "mentionsDegree": (roots, {"Degree"}),
        "mentionsCertification": (roots, {"Certification"}),
    }
    spans = {**mentions, **entries}
    assertion_records = {**claims, **requirements}
    for record in extraction.relations:
        path = f"relations.{record.id}"
        span(record.evidence_start, record.evidence_end, path)
        if record.predicate not in rules:
            issue("relation_predicate", path, "Relation predicate is not supported by the validation contract")
            continue
        domains, ranges = rules[record.predicate]
        if types.get(record.subject) not in domains or types.get(record.object) not in ranges:
            issue("relation_types", path, "Relation references absent endpoints or incompatible subject/object types")
            continue
        for endpoint in (record.subject, record.object):
            target = spans.get(endpoint)
            if target is not None and not covers(record.evidence_start, record.evidence_end, target):
                issue("relation_evidence", path, "Relation evidence does not cover its referenced endpoint")
            assertion = assertion_records.get(endpoint)
            if assertion is not None and not (record.evidence_start <= assertion.evidence_start
                                             and assertion.evidence_end <= record.evidence_end):
                issue("relation_evidence", path, "Relation evidence does not cover its referenced assertion")
        if record.predicate == "aboutMention" and assertion_records[record.subject].mention_id != record.object:
            issue("assertion_relation", path, "aboutMention disagrees with the assertion's mention reference")
        if record.predicate == "inEntry" and claims[record.subject].entry_id != record.object:
            issue("entry_relation", path, "inEntry disagrees with the claim's entry reference")
        if any(endpoint in entries and entries[endpoint].ambiguous for endpoint in (record.subject, record.object)):
            issue("ambiguous_relation", path, "An asserted relationship uses an ambiguous paragraph entry", "warning")
        if record.predicate in {"inRole", "atOrganization", "forRole", "postedBy"} and mentions[record.object].polarity != "positive":
            issue("negative_association", path, "Association uses negated/uncertain evidence and is omitted from the positive ontology view", "warning")
    links = {(r.subject, r.predicate, r.object) for r in extraction.relations}
    for record in [*claims.values(), *requirements.values()]:
        predicate = ("hasRequirement" if record.id in requirements else
                     "hasSkillClaim" if record.kind == "skill" else "hasCredentialClaim")
        if (root, predicate, record.id) not in links or (record.id, "aboutMention", record.mention_id) not in links:
            issue("missing_assertion_relation", f"assertions.{record.id}", "Assertion is missing its subject or mention relationship")
        if record.id in claims and record.entry_id is not None and (record.id, "inEntry", record.entry_id) not in links:
            issue("missing_entry_relation", f"claims.{record.id}", "Claim's entry association is missing its relationship")


def _check_resolutions(extraction, mentions, taxonomy, score, issue):
    seen = set()
    methods = {"exact_alias", "lexical_similarity", "embedding_cosine"}
    for resolution in extraction.resolutions:
        path = f"resolutions.{resolution.mention_id}"
        mention = mentions.get(resolution.mention_id)
        if mention is None or mention.label not in {"skill", "job title"} or resolution.mention_id in seen:
            issue("resolution_reference", path, "Resolution has an absent, incompatible or duplicated mention reference")
            continue
        seen.add(resolution.mention_id)
        kind = "occupation" if mention.label == "job title" else "skill"
        candidates = {}
        for index, candidate in enumerate(resolution.candidates):
            candidate_path = f"{path}.candidates.{index}"
            key = (candidate.namespace, candidate.id)
            if key in candidates:
                issue("duplicate_candidate", candidate_path, "Candidate namespace/ID pair is duplicated")
            candidates[key] = candidate
            reference = taxonomy.records.get(key)
            allowed_namespaces = {"onet:occupation"} if kind == "occupation" else {"onet:element", "local:onet-software"}
            if (reference is None or candidate.namespace not in allowed_namespaces
                    or candidate.version != taxonomy.version or (reference and candidate.label != reference["label"])):
                issue("taxonomy_candidate", candidate_path, "Candidate identity, label, kind or release does not match pinned O*NET data")
            if candidate.method not in methods:
                issue("retrieval_method", candidate_path, "Candidate retrieval method is not supported")
            score(candidate.score, candidate.method, candidate_path + ".score")
            for method, value in candidate.scores.items():
                if method not in methods | {"context_token_overlap"}:
                    issue("retrieval_method", candidate_path + ".scores", "Candidate has an unsupported score method")
                score(value, method, candidate_path + ".scores." + method)
            if candidate.method in candidate.scores and candidate.score != candidate.scores[candidate.method]:
                issue("candidate_score", candidate_path, "Primary score disagrees with the recorded score for its method")
        if resolution.candidate_count < len(resolution.candidates):
            issue("candidate_count", path, "Candidate count is smaller than the returned list")
        selected = (resolution.selected_namespace, resolution.selected_id)
        if resolution.status == "linked_exact":
            exact = taxonomy.aliases[kind].get(match_key(mention.text), set())
            if (selected not in candidates or selected not in exact or len(exact) != 1
                    or len(match_key(mention.text)) < 4 or mention.polarity != "positive"
                    or (selected in candidates and candidates[selected].method != "exact_alias")):
                issue("taxonomy_selection", path, "Selected identity is not a supported, unambiguous exact alias")
        elif resolution.selected_id is not None or resolution.selected_namespace is not None:
            issue("unexpected_selection", path, "An unresolved/candidate/ambiguous result must not select an identity")
        if resolution.status == "unresolved" and resolution.candidates:
            issue("resolution_status", path, "Unresolved status is inconsistent with the candidate list")
        if resolution.status in {"candidate", "ambiguous"} and not resolution.candidates:
            issue("resolution_status", path, "Candidate/ambiguous status requires candidates")
        if resolution.status == "ambiguous" and resolution.candidate_count < 2:
            issue("resolution_status", path, "Ambiguous status requires at least two candidate identities")
        if resolution.status != "linked_exact":
            issue("unresolved_mapping", path, "Taxonomy mapping still needs contextual review", "warning")
    if {m.id for m in mentions.values() if m.label in {"skill", "job title"}} - seen:
        issue("missing_resolution", "resolutions", "One or more skill/role mentions have no normalization outcome")
