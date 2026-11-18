"""O*NET concept identity shared with the parser.

Concept IDs are built exactly as in ``data_extraction.ontology_population``:
``urn:career-coach:concept:<kind>:sha256(json([namespace, id, source_version]))``.
That lets a skill found in a job posting be compared with a resume skill by ID,
and lets older output files that carry no labels be read back into labels.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from career_coach.data_extraction.entity_extraction import TaxonomyResolver


DEFAULT_TAXONOMY_DIR = Path(__file__).resolve().parents[2] / "data/taxonomy/onet/31.0"


def concept_id(kind: str, namespace: str, identifier: str, version: str) -> str:
    """Return the parser's IRI for an O*NET concept (kind is "skill" or "occupation")."""
    digest = hashlib.sha256(json.dumps([namespace, identifier, version], sort_keys=True,
                                       ensure_ascii=False).encode()).hexdigest()
    return f"urn:career-coach:concept:{kind}:{digest}"


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    kind: str          # "skill" or "occupation"
    namespace: str     # e.g. "onet:element", "local:onet-software", "onet:occupation"
    taxonomy_id: str
    version: str


class Taxonomy:
    """In-memory view of the pinned O*NET lookups used by the parser."""

    def __init__(self, taxonomy_dir=DEFAULT_TAXONOMY_DIR):
        self.directory = Path(taxonomy_dir)
        self.resolver = TaxonomyResolver(self.directory)
        self.version = self.resolver.taxonomy["version"]
        self.by_id: dict[str, Concept] = {}
        # alias match_key -> concepts; several concepts may share one alias.
        self.aliases: dict[str, dict[str, list[Concept]]] = {"skill": {}, "occupation": {}}
        for kind, index in self.resolver.index.items():
            for alias, candidates in index.items():
                concepts = [self._concept(kind, c) for c in candidates]
                self.aliases[kind][alias] = concepts
        # Curated parser aliases (e.g. "aws") resolve to the same concepts as the parser.
        for short, target in TaxonomyResolver.SKILL_ALIASES.items():
            concepts = self.aliases["skill"].get(self.match_key(target))
            if concepts and short not in self.aliases["skill"]:
                self.aliases["skill"][short] = concepts

    def _concept(self, kind, candidate) -> Concept:
        identity = concept_id(kind, candidate["namespace"], candidate["id"], candidate["source_version"])
        concept = self.by_id.get(identity)
        if concept is None:
            concept = Concept(identity, candidate["label"], kind, candidate["namespace"],
                              candidate["id"], candidate["source_version"])
            self.by_id[identity] = concept
        return concept

    @staticmethod
    def match_key(text: str) -> str:
        return TaxonomyResolver.match_key(text)

    def label(self, identity: str) -> str | None:
        concept = self.by_id.get(identity)
        return concept.label if concept else None
