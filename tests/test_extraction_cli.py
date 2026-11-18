import copy
import importlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

cli = importlib.import_module("career_coach.data_extraction.main")


class ExtractionOutputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features = [{"cv_id": "resume.pdf", "section_texts": {"skills": "Python"}}]
        self.mapping = {"entity_id": "python-evidence", "status": "resolved", "taxonomy_type": "skill",
                        "entity": {"text": "Python", "label": "programming language", "section": "skills",
                                   "source_field": "section_texts.skills", "start": 0, "end": 6},
                        "mapping": {"id": "python", "namespace": "test:software",
                                    "source_version": "31.0", "label": "Python"}}

    def run_cli(self, rules_only=False, failure=None):
        original = copy.deepcopy(self.features)
        arguments = ["extract", str(self.root / "data")]
        if rules_only:
            arguments.append("--rules-only")
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", arguments))
            parser = stack.enter_context(patch.object(cli, "ResumeParser"))
            extractor = stack.enter_context(patch.object(cli, "EntityExtractor"))
            taxonomy = stack.enter_context(patch.object(cli, "TaxonomyResolver"))
            llm = stack.enter_context(patch("career_coach.data_extraction.llm_mapping.LLMMappingResolver"))
            parsed = Mock()
            parsed.to_dict.return_value = self.features
            parser.return_value.run.return_value = parsed
            extractor.return_value.extract_entities.return_value = []
            taxonomy.return_value.resolve_entities.return_value = {"resolved": []}
            llm.return_value.resolve.return_value = {
                "entity_mappings": {"resolved": [self.mapping]}, "llm_audit": {"calls": 1}}
            if failure == "llm":
                llm.return_value.resolve.side_effect = ValueError("Review failed")
            elif failure == "population":
                stack.enter_context(patch("career_coach.data_extraction.ontology_population.OntologyPopulator.populate",
                                          side_effect=ValueError("Population failed")))
            if failure:
                with self.assertRaises(ValueError):
                    cli.main()
            else:
                cli.main()
            if rules_only:
                extractor.assert_not_called()
                taxonomy.assert_not_called()
                llm.assert_not_called()
            elif not failure:
                llm.return_value.resolve.assert_called_once_with(
                    self.features[0], taxonomy.return_value.resolve_entities.return_value)
        self.assertEqual(self.features, original)

    def test_only_final_pydantic_json_is_written(self):
        self.run_cli()
        self.assertEqual({p.name for p in self.root.iterdir()}, {"data.json"})
        result = json.loads((self.root / "data.json").read_text(encoding="utf-8"))
        self.assertEqual(set(result), {"resumes"})
        resume = result["resumes"][0]
        self.assertEqual(set(resume), {"cv_id", "nodes"})
        self.assertEqual(resume["cv_id"], "resume.pdf")
        nodes = {n["id"]: n for n in resume["nodes"]}
        person = next(n for n in nodes.values() if n["type"].endswith("#Person"))
        skill = nodes[person["hasSkill"][0]]
        self.assertTrue(skill["type"].endswith("#Skill"))
        self.assertEqual(skill["metadata"]["label"], "Python")
        self.assertEqual(skill["metadata"]["taxonomy"], self.mapping["mapping"])
        self.assertEqual(skill["metadata"]["evidence"][0]["text"], "Python")
        self.assertIn("hasExperience", person)
        self.assertNotIn("has_skill", person)

    def test_rules_only_keeps_one_parsing_output(self):
        self.run_cli(rules_only=True)
        self.assertEqual({p.name for p in self.root.iterdir()}, {"data.json"})
        self.assertEqual(json.loads((self.root / "data.json").read_text()), self.features)

    def test_failure_produces_no_partial_or_intermediate_files(self):
        for stage in ("llm", "population"):
            with self.subTest(stage=stage):
                self.run_cli(failure=stage)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_failure_preserves_previous_final_output(self):
        output = self.root / "data.json"
        output.write_text("previous successful output", encoding="utf-8")
        self.run_cli(failure="population")
        self.assertEqual(output.read_text(encoding="utf-8"), "previous successful output")
        self.assertEqual({p.name for p in self.root.iterdir()}, {"data.json"})

    def test_each_resume_runs_through_population_in_order(self):
        self.features = [{"cv_id": name, "section_texts": {}} for name in ("first.pdf", "second.pdf")]
        events = []
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["extract", str(self.root / "data")]))
            parser = stack.enter_context(patch.object(cli, "ResumeParser"))
            extractor = stack.enter_context(patch.object(cli, "EntityExtractor"))
            taxonomy = stack.enter_context(patch.object(cli, "TaxonomyResolver"))
            llm = stack.enter_context(patch("career_coach.data_extraction.llm_mapping.LLMMappingResolver"))
            from career_coach.data_extraction.ontology_population import OntologyPopulator
            actual_populate = OntologyPopulator.populate
            def populate(instance, features, reviewed):
                events.append(("population", features["cv_id"]))
                return actual_populate(instance, features, reviewed)
            stack.enter_context(patch.object(OntologyPopulator, "populate", populate))
            parser.return_value.run.return_value.to_dict.return_value = self.features
            def extract(features):
                events.append(("extract", features["cv_id"]))
                return []
            def resolve(entities, *, document_id):
                events.append(("taxonomy", document_id))
                return {}
            def review(features, mappings):
                events.append(("llm", features["cv_id"]))
                return {"entity_mappings": {}, "llm_audit": {}}
            extractor.return_value.extract_entities.side_effect = extract
            taxonomy.return_value.resolve_entities.side_effect = resolve
            llm.return_value.resolve.side_effect = review
            cli.main()
        self.assertEqual(events, [(stage, name) for name in ("first.pdf", "second.pdf")
                                  for stage in ("extract", "taxonomy", "llm", "population")])
        result = json.loads((self.root / "data.json").read_text())
        self.assertEqual(len(result["resumes"]), 2)
        self.assertEqual({p.name for p in self.root.iterdir()}, {"data.json"})
