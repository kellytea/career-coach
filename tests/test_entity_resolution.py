"""Resolution tests use small, explicit taxonomy fixtures, not model predictions."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path

TaxonomyResolver = importlib.import_module("career_coach.data_extraction.entity_extraction").TaxonomyResolver


class TaxonomyResolverTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "manifest.json").write_text(json.dumps({"version": "31.0"}), encoding="utf-8")
        def row(identity, label, namespace="local:onet-software", aliases=None):
            return {"id": identity, "label": label, "namespace": namespace,
                    "source_version": "31.0", "aliases": aliases or []}
        files = {
            "skills.jsonl": [row("skill-1", "Programming", "onet:element")],
            "software_examples.jsonl": [row(label, label) for label in
                ("Python", "C", "C++", "C#", "Oracle Java", "Microsoft .NET Framework")],
            "occupations.jsonl": [
                row("occupation-1", "Software Developers", "onet:occupation", ["Developer", "Software Engineer"]),
                row("occupation-2", "Web Developers", "onet:occupation", ["Developer"]),
            ],
        }
        for name, rows in files.items():
            (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.resolver = TaxonomyResolver(root)

    def resolve(self, text, label="technical skill"):
        return self.resolver.resolve_entities([
            {"text": text, "label": label, "start": 0, "end": len(text), "section": "skills"}
        ], document_id="resume.pdf")

    def test_exact_and_alias_mappings_preserve_evidence(self):
        exact = self.resolve("PYTHON")["resolved"][0]
        self.assertEqual(exact["mapping"]["label"], "Python")
        self.assertEqual(exact["entity"]["text"], "PYTHON")
        self.assertEqual(exact["method"], "exact_match")
        alias = self.resolve("Java")["resolved"][0]
        self.assertEqual(alias["mapping"]["label"], "Oracle Java")
        self.assertEqual(alias["method"], "curated_alias")
        self.assertEqual(alias["mapping"]["source_version"], "31.0")

    def test_punctuation_is_not_discarded(self):
        ids = [self.resolve(text)["resolved"][0]["mapping"]["id"] for text in ("C", "C++", "C#")]
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(self.resolve(".NET")["resolved"][0]["mapping"]["label"], "Microsoft .NET Framework")
        self.assertEqual(self.resolve("C/C++")["unresolved"][0]["reason"], "no_taxonomy_match")

    def test_ambiguous_occupation_stays_unresolved(self):
        result = self.resolve("Developer", "job title")
        self.assertEqual(result["resolved"], [])
        item = result["unresolved"][0]
        self.assertEqual(item["reason"], "ambiguous_match")
        self.assertEqual(len(item["candidates"]), 2)
        self.assertIsNone(item["mapping"])
        exact = self.resolve("Software Engineer", "job title")["resolved"][0]
        self.assertEqual(exact["mapping"]["namespace"], "onet:occupation")

    def test_unknown_and_out_of_scope_are_distinct(self):
        self.assertEqual(self.resolve("UnknownTool")["counts"]["unresolved"], 1)
        result = self.resolve("Example University", "university")
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(result["not_applicable"][0]["reason"], "entity_type_not_covered_by_onet_lookups")

    def test_stable_identity_and_json_roundtrip(self):
        first, second = self.resolve("Python"), self.resolve("Python")
        self.assertEqual(first["resolved"][0]["entity_id"], second["resolved"][0]["entity_id"])
        self.assertEqual(json.loads(json.dumps(first)), first)

    def test_candidate_retrieval_does_not_automatically_resolve(self):
        candidates = self.resolver.retrieve_candidates("Software Engineer Intern", "occupation")
        self.assertTrue(any(c["id"] == "occupation-1" for c in candidates))
        self.assertEqual(self.resolve("Software Engineer Intern", "job title")["counts"]["unresolved"], 1)
        self.assertEqual(self.resolver.retrieve_candidates("Example University", "university"), [])
