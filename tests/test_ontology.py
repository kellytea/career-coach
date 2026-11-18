"""Local schema and relationship validation; no inference or database access."""

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from career_coach.data_extraction.ontology import load_ontology


class OntologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = load_ontology()

    def test_current_ttl_is_available_as_models(self):
        self.assertEqual(set(self.schema.models), {
            "Degree", "Organization", "Role", "Skill", "DegreeLevel", "Person",
            "Experience", "JobPosting"})
        self.assertEqual(len(self.schema.properties), 13)
        for prop in self.schema.properties.values():
            name = next(name for name, iri in self.schema.classes.items() if iri == prop.domain)
            field = self.schema.models[name].model_fields[prop.field_name]
            self.assertEqual(field.json_schema_extra["rdf_range"], prop.range)

    def test_empty_fields_make_no_assertions_and_lists_are_independent(self):
        person = self.schema.models["Person"]
        first, second = person(id="urn:person:1"), person(id="urn:person:2")
        first.has_skill.append("urn:skill:python")
        self.assertEqual(second.has_skill, [])
        self.assertEqual(second.holds_degree, [])

    def test_python_names_and_ttl_aliases(self):
        person = self.schema.models["Person"](id="urn:person:1", hasSkill=["urn:skill:python"])
        self.assertEqual(person.has_skill, ["urn:skill:python"])
        self.assertEqual(person.model_dump(by_alias=True)["hasSkill"], ["urn:skill:python"])
        self.assertEqual(person.rdf_type, "http://example.org/jobmatch#Person")

    def test_datatype_and_unknown_fields_are_checked(self):
        with self.assertRaises(ValidationError):
            self.schema.models["DegreeLevel"](id="urn:degree:bachelor", rank=["1"])
        with self.assertRaises(ValidationError):
            self.schema.models["Experience"](id="urn:experience:1", yearsExperience=[float("nan")])
        with self.assertRaises(ValidationError):
            self.schema.models["Degree"](id="urn:degree:1", institution=["urn:org:1"])
        with self.assertRaises(ValidationError):
            self.schema.models["Person"](id="not an IRI")

    def test_references_must_exist_and_have_the_declared_type(self):
        person = self.schema.models["Person"](id="urn:person:1", has_skill=["urn:concept:1"])
        skill = self.schema.models["Skill"](id="urn:concept:1")
        self.schema.validate_instances([person, skill])
        with self.assertRaisesRegex(ValueError, "Missing target"):
            self.schema.validate_instances([person])
        wrong_type = self.schema.models["Organization"](id=skill.id)
        with self.assertRaisesRegex(ValueError, "Wrong target type"):
            self.schema.validate_instances([person, wrong_type])
        with self.assertRaisesRegex(ValueError, "Duplicate node ID"):
            self.schema.validate_instances([skill, skill])

    def test_models_follow_ttl_and_unsupported_ranges_fail(self):
        ttl = '''@prefix : <urn:test:> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        :Example a owl:Class .
        :customValue a owl:DatatypeProperty; rdfs:domain :Example; rdfs:range xsd:string .
        '''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "schema.ttl"
            path.write_text(ttl, encoding="utf-8")
            schema = load_ontology(path)
            node = schema.models["Example"](id="urn:example:1", custom_value=["value"])
            self.assertEqual(node.custom_value, ["value"])
            path.write_text(ttl.replace("xsd:string", "xsd:unknown"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported datatype"):
                load_ontology(path)
