"""Stages 2-5. Offsets always address the saved, parser-produced text."""

import hashlib
import io
import json
import re
import zipfile
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

from .models import Block, Claim, Entry, Extraction, Mention, Relation, Requirement
from .taxonomy import Taxonomy


HEADINGS = {
    "experience": {"experience", "work experience", "professional experience", "employment",
                   "employment history", "work history", "professional background"},
    "skills": {"skills", "technical skills", "core competencies", "technologies", "expertise"},
    "education": {"education", "academic background", "qualifications"},
    "projects": {"projects", "personal projects", "selected projects"},
    "certifications": {"certifications", "certificates", "licenses"},
    "requirements": {"requirements", "required qualifications", "minimum qualifications",
                     "what you bring", "what we're looking for", "must have"},
    "preferred": {"preferred qualifications", "preferred skills", "nice to have", "bonus points"},
    "responsibilities": {"responsibilities", "what you'll do", "duties"},
    "overview": {"summary", "profile", "professional summary", "about the role", "description"},
    "title": {"job title"}, "organization": {"company"}, "location": {"location"},
}
LABELS = ["job title", "organization", "skill", "degree", "certification", "location", "date"]
# This extractor is part of the pipeline, not a deployment or request option.
GLINER_MODEL_ID = "urchade/gliner_small-v2.1"
GLINER_REVISION = "4e091416cf7c3481db542c2a3d26156916f3a47f"
GLINER_THRESHOLD = 0.5
ONTOLOGY_NAMESPACE = "http://example.org/jobmatch#"
ENTITY_TYPES = {"job title": "Role", "organization": "Organization", "skill": "Skill", "degree": "Degree"}
MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE = rf"(?:{MONTH}\.?\s+\d{{4}}|(?:19|20)\d{{2}}|present|current)"
DATE_RANGE = re.compile(rf"\b{DATE}\s*(?:[-–—]|\bto\b)\s*{DATE}\b", re.I)
YEARS = re.compile(r"\b\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\+?\s+years?(?:\s+of)?(?:\s+experience)?\b", re.I)
DEGREE = re.compile(r"\b(?:bachelor(?:'s|’s)?|master(?:'s|’s)?|associate(?:'s|’s)?|doctorate|Ph\.?D\.?|B\.?Sc\.?|M\.?Sc\.?|MBA)(?:\s+(?:degree|of\s+(?:science|arts)))?\b", re.I)


class NeedsOCR(ValueError):
    """Native PDF text is absent; source remains archived for later OCR."""


def html_text(payload):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(payload, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    for tag in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "br", "tr", "section"]):
        tag.insert_before("\n")
        tag.insert_after("\n")
    return soup.get_text()


def parse(payload, format):
    warnings = []
    if format == "pdf":
        import pdfplumber
        if not payload.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF signature")
        with pdfplumber.open(io.BytesIO(payload)) as document:
            if len(document.pages) > 100:
                raise ValueError("PDF exceeds 100-page limit")
            pages = [(page.extract_text() or "", i + 1) for i, page in enumerate(document.pages)]
        if any(len(text.strip()) < 10 for text, _ in pages):
            # Fail the whole document rather than silently dropping a scanned page.
            raise NeedsOCR("One or more pages need OCR or manual layout review")
        warnings.append("Native PDF reading order is heuristic; multi-column layouts need review.")
    elif format == "docx":
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 50_000_000:
                raise ValueError("DOCX expanded size exceeds limit")
        document = Document(io.BytesIO(payload))
        parts = []
        for element in document.iter_inner_content():
            if isinstance(element, Paragraph):
                parts.append(element.text)
            elif isinstance(element, Table):
                for row in element.rows:
                    parts.append("\t".join(cell.text for cell in row.cells))
        pages = [("\n".join(parts), None)]
        warnings.append("DOCX body paragraphs/tables only; headers, text boxes and tracked changes need review.")
    elif format == "jobicy_json":
        job = json.loads(payload)
        body = html_text(str(job.get("jobDescription", "")))
        if not body.strip():
            raise ValueError("Jobicy posting has no description")
        pages = [(f"JOB TITLE\n{html_text(str(job['jobTitle']))}\nCOMPANY\n"
                  f"{html_text(str(job.get('companyName', '')))}\nLOCATION\n"
                  f"{html_text(str(job.get('jobGeo', '')))}\nDESCRIPTION\n{body}", None)]
    elif format in {"txt", "html"}:
        decoded = payload.decode("utf-8-sig")
        pages = [(html_text(decoded) if format == "html" else decoded, None)]
    else:
        raise ValueError("Unsupported document format")
    text = ""
    page_spans = []
    for content, page in pages:
        start = len(text)
        text += content + "\n"
        page_spans.append((start, len(text), page))
    if not text.strip():
        raise ValueError("No extractable text")
    if len(text) > 500_000:
        raise ValueError("Extracted text exceeds 500,000-character limit")
    return text, page_spans, warnings


def segment(text, page_spans, kind):
    blocks = []
    section = "overview"
    cursor = 0
    def heading(value):
        key = re.sub(r"\s+", " ", value.casefold().strip(" :•-*\t\r\n"))
        if kind == "job" and key in {"qualifications", "education", "education and experience", "required skills"}:
            return "requirements"
        return next((name for name, variants in HEADINGS.items() if key in variants), None)

    def append(start, end, is_heading=False):
        page = next((p for left, right, p in page_spans if left <= start < right), None)
        blocks.append(Block(id=f"b{len(blocks)}", start=start, end=end, page=page,
                            section=section, heading=is_heading,
                            bullet=bool(re.match(r"\s*(?:[•●▪*-]|\d+[.)])\s+", text[start:end]))))

    for line in text.splitlines(keepends=True):
        # Split inline headings without rewriting text or shifting evidence offsets.
        inline = re.match(r"\s*([^:\r\n]{1,60}):[ \t]*(?=\S)", line)
        inline_section = heading(inline.group(1)) if inline else None
        found = inline_section or heading(line.strip())
        if found:
            section = found
        if inline_section:
            append(cursor, cursor + inline.end(), True)
            append(cursor + inline.end(), cursor + len(line))
        else:
            append(cursor, cursor + len(line), found is not None)
        cursor += len(line)
    return blocks


@lru_cache(maxsize=1)
def gliner_model():
    from gliner import GLiNER
    # Download once on first use, then reuse Hugging Face's disk cache and this
    # worker's in-memory instance. Loading failures propagate; extraction cannot skip NER.
    model = GLiNER.from_pretrained(GLINER_MODEL_ID, revision=GLINER_REVISION)
    model.eval()
    return model


@lru_cache(maxsize=2)
def taxonomy(directory, embedding_path):
    return Taxonomy(directory, embedding_path)


def entities(text, blocks):
    mentions = []
    seen = set()

    def add(label, start, end, method, score=None):
        overlapping = [b for b in blocks if b.start < end and start < b.end]
        if (not overlapping or start < 0 or end > len(text) or start >= end
                or any(b.heading for b in overlapping)
                or len({(b.page, b.section) for b in overlapping}) != 1
                or (label, start, end) in seen):
            return
        block = overlapping[0]
        seen.add((label, start, end))
        prefix = text[max(block.start, start - 100):start].casefold()
        polarity = "negative" if re.search(r"\b(?:no|not|without|never)\b", prefix) else "positive"
        mentions.append(Mention(label=label, ontology_type=ONTOLOGY_NAMESPACE + ENTITY_TYPES[label] if label in ENTITY_TYPES else None,
                                text=text[start:end], start=start, end=end, block_id=block.id,
                                block_ids=[b.id for b in overlapping], method=method, score=score, polarity=polarity))

    for block in blocks:
        if block.heading:
            continue
        content = text[block.start:block.end]
        for label, pattern in (("date range", DATE_RANGE), ("experience duration", YEARS), ("degree", DEGREE)):
            for match in pattern.finditer(content):
                add(label, block.start + match.start(), block.start + match.end(), "regex")
        if block.section in {"title", "organization", "location"} and content.strip():
            start = block.start + len(content) - len(content.lstrip())
            label = {"title": "job title", "organization": "organization", "location": "location"}[block.section]
            add(label, start, start + len(content.strip()), "structured_source")

    model = gliner_model()
    # Token-count bounded windows with overlapping words; deduplication uses full-document offsets.
    words = list(re.finditer(r"\S+", text))
    tokenizer = model.data_processor.transformer_tokenizer
    start_word = 0
    while start_word < len(words):
        stop_word = min(start_word + 160, len(words))
        start = words[start_word].start()
        # Reserve room for label tokens in the uni-encoder context.
        while stop_word > start_word + 1 and len(tokenizer.encode(
                text[start:words[stop_word - 1].end()], add_special_tokens=False)) > 250:
            stop_word -= 1
        end = words[stop_word - 1].end()
        for item in model.predict_entities(text[start:end], LABELS, threshold=GLINER_THRESHOLD):
            left, right = start + item["start"], start + item["end"]
            if text[left:right] == item["text"]:
                add(item["label"], left, right, "gliner", float(item["score"]))
        if stop_word == len(words):
            break
        start_word = max(start_word + 1, stop_word - 30)
    # Prefer deterministic spans when the same label has overlapping model spans.
    ordered = sorted(mentions, key=lambda m: (m.method == "gliner", -(m.end - m.start), m.start))
    kept = []
    for mention in ordered:
        if not any(m.label == mention.label and m.start < mention.end and mention.start < m.end for m in kept):
            kept.append(mention)
    return sorted(kept, key=lambda m: (m.start, m.end, m.label))


def resume_entries(text, blocks, mentions):
    """Bound cross-line associations to one paragraph in one known section."""
    entries, group = [], []
    sections = {"experience": "experience", "education": "education", "projects": "project"}

    def flush():
        if not group:
            return
        ids = {b.id for b in group}
        items = [m for m in mentions if set(m.block_ids).issubset(ids)]
        ambiguous = (sum(m.label == "job title" for m in items) > 1
                     or sum(m.label == "organization" for m in items) > 1
                     or sum(m.label == "date range" for m in items) > 1)
        entries.append(Entry(id=f"entry:{group[0].id}", kind=sections[group[0].section],
                             block_ids=[b.id for b in group], start=group[0].start,
                             end=group[-1].end, ambiguous=ambiguous))
        group.clear()

    for block in blocks:
        if (block.heading or block.section not in sections or not text[block.start:block.end].strip()
                or (group and (block.section != group[-1].section or block.page != group[-1].page))):
            flush()
        if not block.heading and block.section in sections and text[block.start:block.end].strip():
            group.append(block)
    flush()
    return entries


def relationships(text, blocks, mentions, kind):
    relations, claims, requirements = [], [], []
    entries = resume_entries(text, blocks, mentions) if kind == "resume" else []
    by_id = {b.id: b for b in blocks}
    subject = "document_subject" if kind == "resume" else "job_posting"

    def relate(left, predicate, right, start, end, method="same_block_rule", **attributes):
        relations.append(Relation(subject=left, predicate=predicate, object=right,
                                  evidence_start=start, evidence_end=end, method=method, attributes=attributes))

    for mention in mentions:
        block = by_id[mention.block_id]
        end = max(by_id[bid].end for bid in mention.block_ids)
        start, content = block.start, text[block.start:end]
        items = [m for m in mentions if m.block_id == block.id]
        if kind == "resume" and mention.label in {"skill", "degree", "certification"}:
            entry = next((e for e in entries if not e.ambiguous and set(mention.block_ids).issubset(e.block_ids)), None)
            claim = Claim(kind=mention.label, mention_id=mention.id, entry_id=entry.id if entry else None,
                          evidence_start=start, evidence_end=end, polarity=mention.polarity,
                          basis="listed" if block.section == "skills" else "contextual")
            claims.append(claim)
            relate(subject, "hasSkillClaim" if claim.kind == "skill" else "hasCredentialClaim", claim.id, start, end)
            relate(claim.id, "aboutMention", mention.id, start, end)
            if entry:
                relate(claim.id, "inEntry", entry.id, entry.start, entry.end, "section_paragraph_rule")
            continue
        if kind == "job" and mention.label in {"skill", "degree", "certification", "experience duration"}:
            preferred = block.section == "preferred" or bool(re.search(r"\b(preferred|bonus|nice.to.have)\b", content, re.I))
            required = block.section == "requirements" or bool(re.search(r"\b(must|required)\b", content, re.I))
            if required or preferred or block.section == "skills" or mention.label == "experience duration":
                strength = "preferred" if preferred else "required" if required else "unspecified"
                durations = [m for m in items if m.label == "experience duration"]
                duration = durations[0].id if (len(durations) == 1 and mention.label == "skill"
                           and sum(m.label == "skill" for m in items) == 1) else None
                requirement = Requirement(kind="experience" if mention.label == "experience duration" else mention.label,
                                          mention_id=mention.id, evidence_start=start, evidence_end=end,
                                          strength=strength, polarity=mention.polarity,
                                          alternatives_present=bool(re.search(r"\b(or|either)\b", content, re.I)),
                                          duration_mention_id=duration)
                requirements.append(requirement)
                relate(subject, "hasRequirement", requirement.id, start, end)
                relate(requirement.id, "aboutMention", mention.id, start, end)
                continue
        predicate = {"job title": "mentionsRole", "organization": "mentionsOrganization",
                     "location": "mentionsLocation", "skill": "mentionsSkill", "degree": "mentionsDegree",
                     "certification": "mentionsCertification"}.get(mention.label)
        if kind == "job" and mention.method == "structured_source":
            predicate = {"job title": "forRole", "organization": "postedBy", "location": "mentionsLocation"}.get(mention.label, predicate)
        if predicate:
            relate(subject, predicate, mention.id, start, end, polarity=mention.polarity, section=block.section)

    for entry in entries:
        if entry.kind != "experience" or entry.ambiguous:
            continue
        items = [m for m in mentions if set(m.block_ids).issubset(entry.block_ids)]
        titles = [m for m in items if m.label == "job title"]
        organizations = [m for m in items if m.label == "organization"]
        if len(titles) == 1 and len(organizations) == 1:
            relate(subject, "hasExperience", entry.id, entry.start, entry.end, "section_paragraph_rule")
            for predicate, item in [("inRole", titles[0]), ("atOrganization", organizations[0])]:
                relate(entry.id, predicate, item.id, entry.start, entry.end, "section_paragraph_rule")
            for item in items:
                if item.label == "date range":
                    relate(entry.id, "hasDateRange", item.id, entry.start, entry.end, "section_paragraph_rule")
    return entries, claims, requirements, relations


def extract(payload, format, kind, version_id, config):
    text, pages, warnings = parse(payload, format)
    blocks = segment(text, pages, kind)
    mentions = entities(text, blocks)
    normalizer = taxonomy(config["taxonomy_dir"], config.get("embedding_path"))
    resolutions = []
    for mention in mentions:
        if mention.label in {"skill", "job title"}:
            block = next(b for b in blocks if b.id == mention.block_id)
            section_blocks = [b for b in blocks if b.section == block.section and b.page == block.page]
            nearby = [b for b in section_blocks if b.end >= mention.start - 500 and b.start <= mention.end + 500]
            context = text[nearby[0].start:nearby[-1].end]
            resolutions.append(normalizer.resolve(mention, context))
    entries, claims, requirements, relations = relationships(text, blocks, mentions, kind)
    warnings.extend(["Relationships use same-block and section-paragraph rules; ambiguous entries remain unlinked.",
                     "Negation and requirement modality are heuristic and need review.",
                     "LLM resolution and graph publishing are deferred; validation is reported separately."])
    packages = {}
    for name in ("pdfplumber", "python-docx", "beautifulsoup4", "gliner", "rapidfuzz", "sentence-transformers"):
        try:
            packages[name] = package_version(name)
        except PackageNotFoundError:
            pass
    return Extraction(document_version_id=version_id, text=text,
                      text_sha256=hashlib.sha256(text.encode()).hexdigest(), blocks=blocks,
                      mentions=mentions, entries=entries, claims=claims, requirements=requirements, relations=relations,
                      resolutions=resolutions, warnings=warnings,
                      provenance={"extractor": "gliner", "model_id": GLINER_MODEL_ID,
                                  "model_revision": GLINER_REVISION, "threshold": GLINER_THRESHOLD,
                                  "taxonomy_version": normalizer.version,
                                  "taxonomy_manifest_sha256": hashlib.sha256((normalizer.root / "manifest.json").read_bytes()).hexdigest(),
                                  "embedding_model_path": config.get("embedding_path"), "packages": packages,
                                  "ontology_namespace": ONTOLOGY_NAMESPACE,
                                  "ontology_sha256": hashlib.sha256(Path(config.get("ontology_path",
                                      Path(__file__).resolve().parents[3] / "data/ontologies/resume-job.ttl")).read_bytes()).hexdigest(),
                                  "offset_convention": "Python Unicode character offsets, half-open, into text"})
