"""One-off LinkedIn CSV to evidence-backed job-match ontology JSON export.

Only GLiNER proposes skill spans. Column mapping, requirement filtering, concept
IDs, and ontology population use deterministic rules; no LLM or database calls.
"""

import argparse
from bisect import bisect_left, bisect_right
import csv
import hashlib
import json
import os
import re
from functools import lru_cache
from itertools import islice
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_serializer

from career_coach.data_extraction.entity_extraction import TaxonomyResolver
from career_coach.data_extraction.ontology import NodeMetadata, OntologyNode, SourceEvidence, load_ontology


ROOT = Path(__file__).resolve().parents[2]
MODEL = "urchade/gliner_small-v2.1"
VERSION = "linkedin-job-population-v1"
LABELS = ["technical skill", "programming language", "software tool", "soft skill"]

# Section state is used only for relationship decisions, never to truncate NER.
HEADINGS = re.compile(
    r"(?im)(?:^|\n|\b)(?P<heading>preferred qualifications|preferred skills|"
    r"desired qualifications|nice to have|bonus points|minimum qualifications|"
    r"basic qualifications|required qualifications|required skills and abilities|required skills|preferred|required|job skills|must haves|plusses|"
    r"qualifications|requirements|what you bring|what you need|who you are|"
    r"responsibilities|job duties|about the role|role description|position description|experience and education|what you.ll do|what you will do|"
    r"about us|about the company|who we are|what we offer|benefits|"
    r"compensation|equal opportunity)(?:\s*\([^\n)]*\))?(?:\s*[:\n]|(?=\s*$))")
OPTIONAL = re.compile(r"\b(?:preferred|preferably|desirable|desired|optional|bonus|nice.to.have|a plus|an advantage)\b", re.I)
NEGATED = re.compile(r"\b(?:no|not|without|never)\b.{0,45}\b(?:required|necessary|needed)\b|\bnot\s+required\b", re.I)
REQUIREMENT = re.compile(
    r"\b(?:must|required|requires|proficien\w*|knowledge|experience|expertise|"
    r"ability|able|skills?|familiar\w*|competenc\w*|strong|understanding|using|use|utilize)\b", re.I)
DEGREES = [
    ("associate", 1, re.compile(r"\bassociate(?:['’]s|s)?\s+(?:degree|of\s+(?:arts|science))\b", re.I)),
    ("bachelor", 2, re.compile(r"\bbachelor(?:['’]s|s)?(?:\s+degree)?\b|\bB\.?[SA]\.?(?=\s*(?:/|in\b|or\b|degree\b))|\bBSc\b", re.I)),
    ("master", 3, re.compile(r"\bmaster(?:['’]s|s)(?:\s+degree)?\b|\bmaster\s+(?:degree|of\s+(?:arts|science))\b|\bM\.?S\.?(?=\s*(?:/|in\b|or\b|degree\b))|\bMSc\b", re.I)),
    ("doctorate", 4, re.compile(r"\b(?:Ph\.?D\.?|doctorate|doctoral\s+degree)\b", re.I)),
]
EXPERIENCE = re.compile(
    r"(?<![\w.])(?P<low>\d+(?:\.\d+)?)\s*(?:[-–—]|to)?\s*"
    r"(?:\d+(?:\.\d+)?\s*)?(?:\+|or more)?\s*(?P<unit>years?|months?)\b"
    r"(?:['’]\s*|\s+(?:of\s+)?)?(?:(?:professional|relevant|related|prior|"
    r"hands.on|software\s+development|software\s+engineering|programming|"
    r"industry|work)\s+){0,3}experience\b", re.I)


def digest(value):
    # Same canonical-concept digest as resume ontology population.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=64)
def source_view(text):
    """Insert missing list/sentence separators, with a map to original offsets.

    Restrict repairs to recognizable prose boundaries; preserve identifiers such
    as JavaScript, TypeScript, .NET, Node.js, and C++. The raw CSV is unchanged.
    """
    boundaries = {m.start() for m in re.finditer(
        r"(?<=[a-z][.!?])(?=[A-Z][a-z]{2}|[-*•]\s*[A-Z])|"
        r"(?<=[\w).])(?=(?:Experience|Proficiency|Familiarity|Ability|Knowledge|"
        r"Strong|Excellent|Demonstrated|Minimum|Bachelor|Master|Develop|Collaborate|"
        r"Contribute|Databases(?=:)|Languages(?=:)|Solid\s|Software\s+design)\b)", text)}
    for match in re.finditer(
            r"(?:Qualifications|Responsibilities|Requirements|About the Role|"
            r"Compensation|Role Description|Company Description|Position Description)(?=[A-Z0-9])", text):
        boundaries.add(match.end())
    chars, offsets = [], []
    for index, char in enumerate(text):
        if index in boundaries:
            chars.append("\n")
            offsets.append(index)
        chars.append(char)
        offsets.append(index)
    return "".join(chars), offsets


def context(text, start, end):
    """A local sentence/bullet and the last recognized section heading."""
    text, offsets = source_view(text)
    start, end = bisect_right(offsets, start) - 1, bisect_left(offsets, end)
    breaks = list(re.finditer(r"\n|;|[.!?](?=\s+[A-Z]|$)", text))
    # A comma after an explicit qualifier separates mixed clauses such as
    # "Python required, Java preferred" while preserving bare skill lists.
    for separator in re.finditer(r",|\band\b", text):
        previous = max((m.end() for m in breaks if m.end() <= separator.start()), default=0)
        prefix = re.sub(r"\b(?:required|preferred)(?:\s+qualifications|\s+skills(?:\s+and\s+abilities)?)?\s*:",
                        "", text[previous:separator.start()], flags=re.I)
        if re.search(r"\b(?:required|preferred|optional)\b", prefix, re.I):
            breaks.append(separator)
    left = max((m.end() for m in breaks if m.end() <= start), default=0)
    right = min((m.start() for m in breaks if m.start() >= end), default=len(text))
    headings = list(HEADINGS.finditer(text, 0, start))
    heading = headings[-1]["heading"].casefold() if headings else ""
    return text[left:right], heading


def requirement_status(text, start, end, *, field="description"):
    sentence, heading = context(text, start, end)
    if NEGATED.search(sentence):
        return "excluded", "negated_requirement"
    if OPTIONAL.search(sentence):
        return "preferred", "optional_language"
    if heading.startswith(("preferred", "desired", "nice", "bonus", "plusses")):
        # An explicit Required: clause can override a preceding preferred list.
        if not re.search(r"\b(?:must|required)\b", sentence, re.I):
            return "preferred", "optional_section"
    if heading in {"about us", "about the company", "who we are", "what we offer", "benefits", "compensation", "equal opportunity"}:
        return "excluded", "employer_or_benefits_section"
    if re.search(r"\b(?:training (?:is )?provided|we (?:offer|provide)|our (?:clients|customers|supervisors))\b", sentence, re.I):
        return "excluded", "not_an_applicant_requirement"
    if field == "skills_desc" or heading or REQUIREMENT.search(sentence):
        return "required", "qualification_or_role_expectation"
    return "unasserted", "requirement_context_not_established"


class JobPostingDocument(BaseModel):
    """Internal population result; only ontology assertions are exported."""

    model_config = ConfigDict(extra="forbid")
    job_id: str
    source: dict
    extraction: dict
    nodes: list[SerializeAsAny[OntologyNode]] = Field(default_factory=list)

    @field_serializer("nodes")
    def serialize_nodes(self, nodes):
        return [{"type": n.rdf_type, **n.model_dump(mode="json", by_alias=True, exclude_none=True)} for n in nodes]


def ontology_graph(nodes, schema):
    """Project validated instances to flat Neo4j nodes and relationships.

    Class local names become labels, datatype properties stay on nodes, and
    object properties become edges. The flat label property preserves instance
    names (rdfs:label); extraction/source/taxonomy metadata is not exported.
    """
    schema.validate_instances(nodes)
    class_names = {iri: name for name, iri in schema.classes.items()}
    graph_nodes, relationships = [], set()
    for instance in sorted(nodes, key=lambda n: n.id):
        properties = {}
        if instance.metadata.label is not None:
            properties["label"] = instance.metadata.label
        for prop in schema.properties.values():
            if prop.domain != instance.rdf_type:
                continue
            values = getattr(instance, prop.field_name)
            alias = type(instance).model_fields[prop.field_name].alias
            if prop.kind == "object":
                relationships.update((instance.id, alias, target) for target in values)
            elif values:
                properties[alias] = list(values)
        graph_nodes.append({"id": instance.id, "labels": [class_names[instance.rdf_type]],
                            "properties": properties})
    return {"nodes": graph_nodes,
            "relationships": [{"start": start, "type": kind, "end": end}
                              for start, kind, end in sorted(relationships)]}


class JobSkillExtractor:
    """Batched overlapping windows, preserving original column offsets."""

    def __init__(self, model_name=MODEL, threshold=0.5, batch_size=16, *, model=None, offline=False):
        if not 0 < threshold <= 1 or batch_size < 1:
            raise ValueError("threshold must be in (0, 1] and batch_size must be positive")
        self.model_name, self.threshold, self.batch_size = model_name, threshold, batch_size
        self.model_revision = None
        if model is None:
            if offline:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from gliner import GLiNER
            from huggingface_hub import try_to_load_from_cache
            model = GLiNER.from_pretrained(model_name, local_files_only=offline)
            cached = try_to_load_from_cache(model_name, "gliner_config.json")
            if isinstance(cached, str):
                self.model_revision = Path(cached).parent.name
        self.model = model.eval()

    def windows(self, text):
        tokens = list(self.model.data_processor.words_splitter(text))
        size = min(200, int(self.model.config.max_len))
        overlap = min(max(32, int(self.model.config.max_width) - 1), size - 1)
        tokenizer = self.model.data_processor.transformer_tokenizer
        first = 0
        while first < len(tokens):
            last = min(first + size, len(tokens))
            # Check the actual prompted wordpiece length too: word counts alone
            # can exceed the encoder limit for code/URLs and truncate the tail.
            while True:
                words = [t[0] for t in tokens[first:last]]
                prompted, _ = self.model.data_processor.prepare_inputs([words], LABELS)
                length = len(tokenizer(prompted[0], is_split_into_words=True,
                                       add_special_tokens=True, truncation=False, verbose=False)["input_ids"])
                limit = min(int(tokenizer.model_max_length), int(self.model.config.encoder_config.max_position_embeddings))
                if length <= limit:
                    break
                if last - first == 1:
                    raise ValueError("One source token exceeds the GLiNER encoder budget")
                last = first + max(1, (last - first) // 2)
            start, end = tokens[first][1], tokens[last - 1][2]
            yield start, text[start:end]
            if last == len(tokens):
                break
            first = max(first + 1, last - overlap)

    def extract(self, rows):
        results = [dict() for _ in rows]
        pending = []

        def flush():
            if not pending:
                return
            predictions = self.model.inference([p[3] for p in pending], LABELS,
                                              threshold=self.threshold, batch_size=self.batch_size)
            if len(predictions) != len(pending):
                raise ValueError("GLiNER returned the wrong number of windows")
            for (index, field, offset, window, offsets), entities in zip(pending, predictions):
                for entity in entities:
                    first, last = offset + entity["start"], offset + entity["end"]
                    if not 0 <= entity["start"] < entity["end"] <= len(window) or window[entity["start"]:entity["end"]] != entity["text"]:
                        raise ValueError("GLiNER evidence does not match its source column")
                    start, end = offsets[first], offsets[last - 1] + 1
                    item = {**entity, "text": rows[index][field][start:end], "source_field": field, "start": start, "end": end}
                    key = (field, start, end)
                    previous = results[index].get(key)
                    if previous is None or (item["score"], item["label"]) > (previous["score"], previous["label"]):
                        results[index][key] = item
            pending.clear()

        for index, row in enumerate(rows):
            for field in ("description", "skills_desc"):
                view, offsets = source_view(row.get(field, ""))
                for offset, window in self.windows(view):
                    pending.append((index, field, offset, window, offsets))
                    if len(pending) == self.batch_size:
                        flush()
        flush()
        return [sorted(items.values(), key=lambda e: (e["source_field"], e["start"], e["end"])) for items in results]


class JobPopulator:
    def __init__(self, *, schema=None, taxonomy_dir=ROOT / "data/taxonomy/onet/31.0"):
        self.schema = schema or load_ontology()
        self.resolver = TaxonomyResolver(taxonomy_dir)

    def populate(self, row, entities, *, row_number, extraction):
        job_id = row.get("job_id", "").strip()
        if not re.fullmatch(r"\d+", job_id):
            raise ValueError(f"Row {row_number}: missing or invalid LinkedIn job_id")
        nodes = {}
        source_document = f"linkedin_job_postings:{job_id}"

        def evidence(field, start, end, label, method):
            text = row[field][start:end]
            if not text or not 0 <= start < end <= len(row[field]):
                raise ValueError("Invalid source evidence")
            return {"entity_id": digest([source_document, field, start, end, label]),
                    "source_document": source_document, "source_field": field,
                    "start": start, "end": end, "text": text, "label": label,
                    "extraction_sources": [method]}

        def node(kind, identity, label, supports=(), **metadata):
            if identity not in nodes:
                nodes[identity] = self.schema.models[kind](id=identity,
                    metadata=NodeMetadata(label=label, **metadata))
            instance = nodes[identity]
            for support in supports:
                if all(e.entity_id != support["entity_id"] for e in instance.metadata.evidence):
                    instance.metadata.evidence.append(SourceEvidence(**support))
            return instance

        def concept(kind, text, supports):
            # Reuse the resume resolver's exact/curated alias policy and IDs.
            item = {"text": text, "label": "technical skill" if kind == "Skill" else "job title"}
            resolved = self.resolver.resolve_entities([item], document_id=source_document)
            if resolved["resolved"]:
                mapping = resolved["resolved"][0]["mapping"]
                identity = "urn:career-coach:concept:" + ("skill" if kind == "Skill" else "occupation") + ":" + digest(
                    [mapping["namespace"], mapping["id"], mapping["source_version"]])
                return node(kind, identity, mapping["label"], supports, taxonomy=mapping,
                            resolution_status="resolved")
            return node(kind, f"urn:career-coach:local-{kind.lower()}:" + digest(self.resolver.match_key(text)),
                        text, supports, resolution_status="unresolved",
                        resolution_reason=resolved["unresolved"][0]["reason"])

        job = node("JobPosting", f"urn:career-coach:linkedin:job:{job_id}", row.get("title") or job_id)
        for field, kind in (("title", "Role"), ("company_name", "Organization")):
            raw = row.get(field, "")
            if not raw.strip():
                continue
            start, end = len(raw) - len(raw.lstrip()), len(raw.rstrip())
            supports = [evidence(field, start, end, kind, "column")]
            if kind == "Role":
                target = concept(kind, raw.strip(), supports)
                target.metadata.original_title = raw.strip()
                job.for_role = [target.id]
            else:
                company_id = row.get("company_id", "").strip()
                if company_id and not re.fullmatch(r"\d+(?:\.0+)?", company_id):
                    raise ValueError(f"Invalid company_id: {company_id}")
                # Unknown IDs use a posting-scoped identity; names need not be unique.
                identity = (f"urn:career-coach:linkedin:company:{company_id.split('.')[0]}" if company_id else job.id + ":organization")
                target = node(kind, identity, raw.strip(), supports)
                job.posted_by = [target.id]

        mentions = []
        for entity in entities:
            field, start, end = entity["source_field"], entity["start"], entity["end"]
            if (field not in {"description", "skills_desc"} or not 0 <= start < end <= len(row.get(field, ""))
                    or row[field][start:end] != entity["text"] or entity["label"] not in LABELS):
                raise ValueError("Skill prediction does not match source")
            status, reason = requirement_status(row[field], start, end, field=field)
            sentence, _ = context(row[field], start, end)
            term = entity["text"].strip().casefold()
            if term in {"skill", "skills", "tool", "tools", "education", "experience", "rock solid experience", "bs", "ms"} or re.search(r"\b(?:engineers?|developers?|designers?)$", term):
                status, reason = "excluded", "not_a_specific_skill"
            elif term in {"computer science", "computer engineering", "engineering", "software", "math"} and any(p.search(sentence) for _, _, p in DEGREES):
                status, reason = "excluded", "education_subject_not_skill"
            mention = {**entity, "status": status, "reason": reason}
            if status == "required":
                target = concept("Skill", entity["text"], [evidence(field, start, end, entity["label"], "gliner")])
                if target.id not in job.requires_skill:
                    job.requires_skill.append(target.id)
                mention["node_id"] = target.id
            mentions.append(mention)

        experience, degree_candidates, requirement_audit = [], {}, []
        for field in ("description", "skills_desc"):
            text = row.get(field, "")
            view, offsets = source_view(text)
            for match in EXPERIENCE.finditer(view):
                start, end = offsets[match.start()], offsets[match.end() - 1] + 1
                status, reason = requirement_status(text, start, end, field=field)
                years = float(match["low"]) / (12 if match["unit"].casefold().startswith("month") else 1)
                sentence, _ = context(text, start, end)
                # Degree-dependent alternatives and maxima cannot become one minimum.
                ambiguous = re.search(r"\bor\b(?!\s+more\b)|\bequivalent\b|\b(?:up to|at most|maximum)\b", sentence, re.I)
                accepted = status == "required" and not ambiguous
                requirement_audit.append({"property": "minExperience", "source_field": field,
                    "start": start, "end": end, "text": text[start:end],
                    "accepted": accepted, "reason": "alternative_or_upper_bound" if ambiguous else reason})
                if accepted:
                    experience.append(years)
                    job.metadata.evidence.append(SourceEvidence(**evidence(field, start, end, "minExperience", "rules")))
            for name, rank, pattern in DEGREES:
                for match in pattern.finditer(view):
                    start, end = offsets[match.start()], offsets[match.end() - 1] + 1
                    status, reason = requirement_status(text, start, end, field=field)
                    sentence, _ = context(text, start, end)
                    alternative = re.search(r"\b(?:equivalent|in lieu|instead)\b|\bor\s+(?:\w+\s+){0,3}experience\b", sentence, re.I)
                    accepted = status == "required" and not alternative
                    requirement_audit.append({"property": "minDegreeLevel", "source_field": field,
                        "start": start, "end": end, "text": text[start:end],
                        "accepted": accepted, "reason": "degree_experience_alternative" if alternative else reason})
                    if accepted:
                        degree_candidates.setdefault((field, sentence), []).append(
                            (rank, name, evidence(field, start, end, "DegreeLevel", "rules")))
        # Distinct experience clauses may concern different skills. A single
        # scalar would lose that scope, so abstain when the values disagree.
        if len(set(experience)) == 1:
            job.min_experience = [experience[0]]
        if degree_candidates:
            # Choose the lower level within an explicit degree alternative,
            # then the highest of independently required qualification clauses.
            requirements = [(min if re.search(r"\bor\b|/", sentence, re.I) else max)(
                                candidates, key=lambda x: (x[0], x[1]))
                            for (_, sentence), candidates in degree_candidates.items()]
            rank, name, support = max(requirements, key=lambda x: (x[0], x[1]))
            level = node("DegreeLevel", f"urn:career-coach:degree-level:v1:{name}", name, [support])
            level.rank = [rank]
            job.min_degree_level = [level.id]

        job.requires_skill.sort()
        ordered = [nodes[key] for key in sorted(nodes)]
        self.schema.validate_instances(ordered)
        return JobPostingDocument(job_id=job_id,
            source={"dataset": "linkedin_job_postings", "row_number": row_number, "fields": dict(row)},
            extraction={**extraction, "population_version": VERSION,
                        "schema_sha256": hashlib.sha256(self.schema.source.read_bytes()).hexdigest(),
                        "omissions": ["minExperience: conflicting or differently scoped values"] if len(set(experience)) > 1 else [],
                        "skill_mentions": mentions, "requirement_mentions": requirement_audit},
            nodes=ordered)


def export_jobs(input_path, output_dir, *, extractor, populator=None, limit=None, title_pattern=None):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    populator = populator or JobPopulator()
    pattern = re.compile(title_pattern, re.I) if title_pattern else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen, count = set(), 0
    config = {"model": extractor.model_name, "model_revision": extractor.model_revision,
              "threshold": extractor.threshold, "labels": LABELS}
    with Path(input_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"job_id", "company_name", "title", "description"} <= set(reader.fieldnames or []):
            raise ValueError("CSV must contain job_id, company_name, title, description")
        def selected():
            for number, row in enumerate(reader, 1):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV record at row {number}")
                if pattern is None or pattern.search(row["title"]):
                    yield number, row
        rows = islice(selected(), limit)
        while batch := list(islice(rows, 16)):
            predictions = extractor.extract([row for _, row in batch])
            if len(predictions) != len(batch):
                raise ValueError("Extractor returned the wrong number of postings")
            for (number, row), entities in zip(batch, predictions):
                document = populator.populate(row, entities, row_number=number, extraction=config)
                if document.job_id in seen:
                    raise ValueError(f"Duplicate job_id: {document.job_id}")
                seen.add(document.job_id)
                target = output_dir / f"{document.job_id}.json"
                temporary = target.with_suffix(".json.tmp")
                graph = ontology_graph(document.nodes, populator.schema)
                temporary.write_text(json.dumps(graph, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
                temporary.replace(target)
                count += 1
                print(f"[{count}] {document.job_id}: {row['title']}", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=ROOT / "data/postings.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/job_postings")
    parser.add_argument("--limit", type=int, help="Maximum matching postings; omit for all rows")
    parser.add_argument("--title-pattern", help="Case-insensitive regular expression for job titles")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--offline", action="store_true", help="Use cached GLiNER model files only")
    args = parser.parse_args()
    extractor = JobSkillExtractor(threshold=args.threshold, batch_size=args.batch_size, offline=args.offline)
    count = export_jobs(args.input, args.output, extractor=extractor, limit=args.limit, title_pattern=args.title_pattern)
    print(f"Wrote {count} validated job posting JSON files to {args.output.resolve()}")


if __name__ == "__main__":
    main()
