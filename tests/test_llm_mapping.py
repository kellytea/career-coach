"""Adversarial LLM-output and grounding checks; no network is used."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from threading import Barrier
from unittest.mock import Mock, patch

from career_coach.data_extraction.entity_extraction import TaxonomyResolver
from career_coach.data_extraction.llm_mapping import LLMMappingError, LLMMappingResolver
from career_coach.data_extraction.parser import ResumeParser


def edit(action, quote=None, label=None, entity_id=None, occurrence=None):
    return {"action": action, "entity_id": entity_id, "quote": quote,
            "occurrence": occurrence, "label": label, "reason": "Explicit evidence in the supplied block"}


class LLMMappingTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"VOYAGER_MODEL": "deepseek-v4-1-flash"})
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.directory = directory
        (directory / "manifest.json").write_text(json.dumps({"version": "31.0"}), encoding="utf-8")
        def row(identity, label, namespace, aliases=None):
            return {"id": identity, "label": label, "namespace": namespace,
                    "source_version": "31.0", "aliases": aliases or []}
        tables = {"skills.jsonl": [], "software_examples.jsonl": [row("python", "Python", "local:onet-software")],
                  "occupations.jsonl": [row("developer", "Software Developers", "onet:occupation", ["Software Engineer"])]}
        for name, rows in tables.items():
            (directory / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.taxonomy = TaxonomyResolver(directory)
        parser = ResumeParser()
        self.features = parser.extract_features(
            "Jane Doe\nEducation\nExample University\nComputer Science, B.S.\n2020 - 2024\n"
            "Computer Science, M.S.\n2024 - 2026\nRelevant Coursework: Operating Systems\nAwards: Example Award\n"
            "Experience\nExample Corp\n2024 - Present\nSoftware Engineer Intern\nBuilt services.\nSkills: Python"
        )
        self.features["cv_id"] = "sample.pdf"
        self.features["education_refined"] = parser.refine_education({"education": self.features["section_texts"]["education"]})
        self.features["experience_refined"] = parser.refine_experience(self.features)
        self.entities = [self.entity("education", "B.S.", "academic major"),
                         self.entity("education", "Example University", "university"),
                         self.entity("experience", "Software Engineer Intern", "job title"),
                         self.entity("skills", "Python", "programming language")]
        self.mappings = self.taxonomy.resolve_entities(self.entities, document_id="sample.pdf")

    def entity(self, section, quote, label):
        text = self.features["section_texts"][section]
        start = text.index(quote)
        return {"text": quote, "start": start, "end": start + len(quote), "label": label,
                "section": section, "source_field": f"section_texts.{section}", "score": 0.7,
                "category": section, "block_index": 0}

    def baseline_item(self, quote):
        return next(item for group in ("resolved", "unresolved", "not_applicable")
                    for item in self.mappings[group] if item["entity"]["text"] == quote)

    def requester(self, request):
        if request["task"] == "select_taxonomy_candidates":
            return {"decisions": [{"query_id": query["query_id"], **self.requester({**query, "task": "select_taxonomy_candidate"})}
                                  for query in request["queries"]]}
        if request["task"] == "select_taxonomy_candidate":
            candidate = next((c for c in request["candidates"] if c["label"] == "Software Developers"), None)
            return {"candidate_id": candidate["candidate_id"] if candidate else None, "reason": "Explicit occupation match"}
        edits = []
        for item in request["eligible_entities"]:
            label = "degree" if item["entity"]["text"] == "B.S." else item["entity"]["label"]
            action = "reclassify" if label != item["entity"]["label"] else "confirm"
            edits.append(edit(action, item["entity"]["text"], label, item["entity_id"]))
        if "M.S." in request["chunk"]["text"]:
            edits.append(edit("add", "M.S.", "degree"))
        return {"edits": edits}

    def resolve(self, requester=None):
        handler = requester or self.requester
        def adapter(request):
            if requester is not None and request["task"] == "select_taxonomy_candidates":
                return {"decisions": [{"query_id": query["query_id"], **handler({**query, "task": "select_taxonomy_candidate"})}
                                      for query in request["queries"]]}
            return handler(request)
        resolver = LLMMappingResolver(self.taxonomy, requester=adapter)
        return resolver.resolve(self.features, self.mappings)

    def test_repairs_explicit_entities_and_selects_real_taxonomy_candidate(self):
        before = copy.deepcopy((self.features, self.mappings))
        result = self.resolve()
        self.assertEqual((self.features, self.mappings), before)
        revised = result["entity_mappings"]
        degree_items = [i for i in revised["not_applicable"] if i["entity"]["label"] == "degree"]
        self.assertEqual({i["entity"]["text"] for i in degree_items}, {"B.S.", "M.S."})
        corrected = next(i for i in degree_items if i["entity"]["text"] == "B.S.")
        self.assertEqual(corrected["llm_review"]["original_entity"]["label"], "academic major")
        self.assertIsNone(corrected["entity"]["score"])
        university = next(i for i in revised["not_applicable"] if i["entity"]["label"] == "university")
        self.assertEqual(university["llm_review"]["action"], "confirm")
        self.assertIsNone(university["mapping"])
        occupation = next(i for i in revised["resolved"] if i["taxonomy_type"] == "occupation")
        self.assertEqual(occupation["mapping"]["id"], "developer")
        self.assertEqual(occupation["method"], "llm_candidate_selection")
        self.assertEqual(next(i for i in revised["resolved"] if i["entity"]["text"] == "Python"), self.baseline_item("Python"))
        for group in ("resolved", "unresolved", "not_applicable"):
            for item in revised[group]:
                entity = item["entity"]
                self.assertEqual(self.features["section_texts"][entity["section"]][entity["start"]:entity["end"]], entity["text"])
        self.assertEqual(sum(revised["counts"].values()), 5)

    def test_coursework_and_awards_are_not_submitted_for_inference(self):
        requests = []
        def requester(request):
            requests.append(request)
            return self.requester(request)
        self.resolve(requester)
        for request in requests:
            if request["task"] == "review_explicit_entities":
                self.assertNotIn("Operating Systems", request["chunk"]["text"])
                self.assertNotIn("Example Award", request["chunk"]["text"])

    def test_unknown_quotes_ids_and_out_of_scope_labels_are_rejected(self):
        for proposal in (edit("add", "Invented Degree", "degree"),
                         edit("reclassify", "B.S.", "degree", "invented-id"),
                         edit("add", "Computer Science", "technical skill")):
            with self.subTest(proposal=proposal):
                def requester(request):
                    if request["task"] == "review_explicit_entities":
                        return {"edits": [proposal] if "B.S." in request["chunk"]["text"] else []}
                    return {"candidate_id": None, "reason": "Uncertain"}
                result = self.resolve(requester)
                self.assertTrue(result["llm_audit"]["rejected"])
                self.assertEqual(sum(result["entity_mappings"]["counts"].values()), 4)

    def test_resolved_entities_cannot_be_changed_or_duplicated(self):
        item = self.baseline_item("Python")
        for proposal in (edit("reclassify", "Python", "software tool", item["entity_id"]),
                         edit("add", "Python", "software tool")):
            def requester(request):
                if request["task"] == "review_explicit_entities":
                    return {"edits": [proposal] if request["chunk"]["section"] == "skills" else []}
                return {"candidate_id": None, "reason": "Uncertain"}
            result = self.resolve(requester)
            self.assertTrue(result["llm_audit"]["rejected"])
            self.assertEqual(result["entity_mappings"]["resolved"][0], item)

    def test_fabricated_candidate_id_never_resolves(self):
        def requester(request):
            if request["task"] == "review_explicit_entities":
                return {"edits": []}
            return {"candidate_id": "invented-onet-id", "reason": "Bad selection"}
        result = self.resolve(requester)
        self.assertEqual(result["entity_mappings"]["counts"]["unresolved"], 1)
        self.assertTrue(result["llm_audit"]["rejected"])

    def test_no_candidate_search_result_means_no_taxonomy_call(self):
        self.taxonomy.index["occupation"].clear()
        requests = []
        def requester(request):
            requests.append(request["task"])
            return {"edits": []}
        result = self.resolve(requester)
        self.assertNotIn("select_taxonomy_candidate", requests)
        self.assertEqual(result["llm_audit"]["taxonomy_decisions"][0]["status"], "no_candidates")

    def test_extra_relationship_keys_and_bad_types_fail_closed(self):
        for response in ({"edits": [], "relationships": []},
                         {"edits": [{**edit("add", "M.S.", "degree"), "occurrence": True}]},
                         "```json\n{}\n```"):
            with self.subTest(response=response):
                with self.assertRaises(LLMMappingError):
                    self.resolve(lambda request: response)

    def test_call_budget_and_stale_evidence_fail_before_mutating_inputs(self):
        with patch("career_coach.data_extraction.llm_mapping.MAX_CALLS", 1):
            with self.assertRaises(LLMMappingError):
                self.resolve()
        self.features["section_texts"]["education"] = "Stale input"
        with self.assertRaises(ValueError):
            self.resolve()

    def test_repeated_missing_quotes_require_an_occurrence(self):
        self.features = {"cv_id": "repeat.pdf", "section_texts": {"skills": "Node.js, Node.js"}}
        self.mappings = self.taxonomy.resolve_entities([], document_id="repeat.pdf")
        result = self.resolve(lambda request: {"edits": [edit("add", "Node.js", "runtime")]})
        self.assertEqual(result["entity_mappings"]["counts"]["unresolved"], 0)
        self.assertTrue(result["llm_audit"]["rejected"])

    def test_explicit_occurrence_is_mapped_to_the_exact_source_position(self):
        self.features = {"cv_id": "repeat.pdf", "section_texts": {"skills": "Node.js, Node.js"}}
        self.mappings = self.taxonomy.resolve_entities([], document_id="repeat.pdf")
        result = self.resolve(lambda request: {"edits": [edit("add", "Node.js", "runtime", occurrence=1)]})
        entity = result["entity_mappings"]["unresolved"][0]["entity"]
        self.assertEqual(entity["start"], 9)
        self.assertEqual(entity["end"], 16)

    def test_reclassification_cannot_move_evidence_to_another_degree(self):
        original = self.baseline_item("B.S.")
        def requester(request):
            if request["task"] == "review_explicit_entities":
                return {"edits": [edit("reclassify", "M.S.", "degree", original["entity_id"])] if "B.S." in request["chunk"]["text"] else []}
            return {"candidate_id": None, "reason": "Uncertain"}
        result = self.resolve(requester)
        item = next(i for i in result["entity_mappings"]["not_applicable"] if i["entity"]["text"] == "B.S.")
        self.assertEqual(item["entity"]["label"], "academic major")
        self.assertTrue(result["llm_audit"]["rejected"])

    def test_model_is_read_from_environment(self):
        with patch.dict(os.environ, {"VOYAGER_MODEL": "another-model"}):
            resolver = LLMMappingResolver(self.taxonomy, requester=self.requester)
        self.assertEqual(resolver.model, "another-model")

    def test_missing_model_requires_environment_configuration(self):
        with patch.dict(os.environ, {"VOYAGER_MODEL": ""}):
            with self.assertRaisesRegex(LLMMappingError, "VOYAGER_MODEL"):
                LLMMappingResolver(self.taxonomy, requester=self.requester)

    def test_transport_uses_json_mode_and_rejects_incomplete_output(self):
        client = Mock()
        create = client.chat.completions.create
        create.return_value = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"edits": []}', refusal=None))])
        # No input mappings means no taxonomy-selection calls in this transport check.
        features = {"cv_id": "test.pdf", "section_texts": {"skills": "Python"}}
        mappings = self.taxonomy.resolve_entities([], document_id="test.pdf")
        resolver = LLMMappingResolver(self.taxonomy, client=client)
        resolver.resolve(features, mappings)
        self.assertEqual(create.call_args.kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(create.call_args.kwargs["reasoning_effort"], "none")
        create.return_value.choices[0].finish_reason = "length"
        with self.assertRaises(LLMMappingError):
            resolver.resolve(features, mappings)

    def test_taxonomy_batch_is_scoped_per_query(self):
        self.features = {"cv_id": "jobs.pdf", "section_texts": {"experience": "Software Engineer Intern; Student Software Engineer"}}
        self.mappings = self.taxonomy.resolve_entities([
            self.entity("experience", "Software Engineer Intern", "job title"),
            self.entity("experience", "Student Software Engineer", "job title"),
        ], document_id="jobs.pdf")
        requests = []
        def requester(request):
            requests.append(request)
            if request["task"] == "review_explicit_entities":
                return {"edits": []}
            first, second = request["queries"]
            return {"decisions": [
                {"query_id": first["query_id"], "candidate_id": second["candidates"][0]["candidate_id"], "reason": "Cross-query ID must be rejected"},
                {"query_id": second["query_id"], "candidate_id": second["candidates"][0]["candidate_id"], "reason": "Same occupation"},
            ]}
        result = LLMMappingResolver(self.taxonomy, requester=requester).resolve(self.features, self.mappings)
        self.assertEqual(result["llm_audit"]["calls"], 2)
        self.assertEqual(result["entity_mappings"]["counts"]["resolved"], 1)
        self.assertEqual(result["entity_mappings"]["counts"]["unresolved"], 1)
        self.assertTrue(result["llm_audit"]["rejected"])
        self.assertEqual(len(requests[-1]["queries"]), 2)

    def test_context_identical_repeated_mentions_share_taxonomy_decisions(self):
        quote = "Software Engineer Intern"
        self.features = {"cv_id": "repeat.pdf", "section_texts": {"experience": quote + "; " + quote}}
        first = self.entity("experience", quote, "job title")
        second = {**first, "start": len(quote) + 2, "end": len(quote) * 2 + 2}
        self.mappings = self.taxonomy.resolve_entities([first, second], document_id="repeat.pdf")
        def requester(request):
            if request["task"] == "review_explicit_entities":
                return {"edits": []}
            self.assertEqual(len(request["queries"]), 1)
            query = request["queries"][0]
            return {"decisions": [{"query_id": query["query_id"], "candidate_id": query["candidates"][0]["candidate_id"], "reason": "Same occupation"}]}
        reviewer = LLMMappingResolver(self.taxonomy, requester=requester)
        result = reviewer.resolve(self.features, self.mappings)
        self.assertEqual(result["entity_mappings"]["counts"]["resolved"], 2)
        self.assertEqual(result["llm_audit"]["taxonomy_cache_hits"], 1)
        again = reviewer.resolve(self.features, self.mappings)
        self.assertEqual(again["llm_audit"]["calls"], 1)
        self.assertEqual(again["llm_audit"]["taxonomy_cache_hits"], 2)

    def test_review_requests_are_parallel_and_timed(self):
        features = {"cv_id": "parallel.pdf", "section_texts": {"title": "Jane Doe", "summary": "Builds reliable services.", "skills": "Python"}}
        barrier = Barrier(3)
        def requester(request):
            barrier.wait(timeout=3)
            return {"edits": []}
        result = LLMMappingResolver(self.taxonomy, requester=requester).resolve(features, self.taxonomy.resolve_entities([], document_id="parallel.pdf"))
        self.assertEqual(result["llm_audit"]["calls"], 3)
        self.assertEqual(len(result["llm_audit"]["timing"]["requests"]), 3)
        self.assertGreaterEqual(result["llm_audit"]["timing"]["total_seconds"], 0)

    def test_resolved_ids_are_not_sent_to_the_model(self):
        requests = []
        def requester(request):
            requests.append(request)
            return self.requester(request)
        result = self.resolve(requester)
        resolved_id = self.baseline_item("Python")["entity_id"]
        for request in requests:
            if request["task"] == "review_explicit_entities":
                self.assertNotIn(resolved_id, json.dumps(request))

    def test_only_documented_conflicts_make_resolved_entities_reviewable(self):
        item = self.baseline_item("Python")
        item["entity"]["classification_conflicts"] = ["organization"]
        requests = []
        def requester(request):
            requests.append(request)
            return self.requester(request)
        result = self.resolve(requester)
        skills_request = next(r for r in requests if r["task"] == "review_explicit_entities" and r["chunk"]["section"] == "skills")
        self.assertEqual(skills_request["eligible_entities"][0]["entity_id"], item["entity_id"])
        self.assertEqual(skills_request["eligible_entities"][0]["entity"]["classification_conflicts"], ["organization"])
        revised = next(i for i in result["entity_mappings"]["resolved"] if i["entity"]["text"] == "Python")
        self.assertNotIn("classification_conflicts", revised["entity"])
