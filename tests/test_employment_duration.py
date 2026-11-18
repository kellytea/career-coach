"""Employment-only month arithmetic, scoped source evidence and ambiguity checks."""

import copy
import json
import unittest
from datetime import date

from career_coach.data_extraction.ontology import OntologyDocument
from career_coach.data_extraction.ontology_population import OntologyPopulator, employment_duration


REFERENCE = date(2026, 11, 17)


class MonthRangeTests(unittest.TestCase):
    def test_complete_named_shared_year_and_numeric_ranges(self):
        cases = {
            "January 2020 - May 2024": 52,
            "Jan. 2020 to May 2024": 52,
            "June – August 2025": 2,
            "June 2025 - August": 2,
            "01/2020 — 05/2024": 52,
            "2020-01 - 2024-05": 52,
            "September 2024 - Sept. 2024": 0,
            "May 2024 – Present": 30,
            "May 2024 - Current": 30,
            "May 2024 \ufffd Present": 30,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                info, reason = employment_duration("Dates: " + text, 100, REFERENCE)
                self.assertIsNone(reason)
                self.assertEqual(info["months"], expected)
                self.assertEqual(info["text"], text)
                self.assertEqual(info["start"], 107)
                self.assertEqual(info["end"], 107 + len(text))
                self.assertEqual(info["as_of_month"], "2026-11")

    def test_incomplete_reversed_future_and_multiple_ranges_abstain(self):
        cases = {
            "September - Present": "missing_year",
            "June - August": "missing_year",
            "2020 - 2024": "missing_month_precision",
            "2020 - Present": "missing_month_precision",
            "December 2025 - January 2025": "reversed_date_range",
            "June - August 2027": "future_or_invalid_date_range",
            "January 2020 - May 2021 | January 2022 - May 2023": "ambiguous_date_ranges",
            "No employment dates": "no_month_range",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                info, reason = employment_duration(text, 0, REFERENCE)
                self.assertIsNone(info)
                self.assertEqual(reason, expected)


class ScopedDurationTests(unittest.TestCase):
    def populate(self, ranges, education="Bachelor, January 2000 - December 2009", body_range=False, missing_ner=False):
        text, blocks, entities = "", [], []
        for index, date_range in enumerate(ranges):
            start = len(text)
            block = f"Employer {index}\nEngineer\n{date_range}\nBuilt services.\n"
            if body_range:
                block = f"Employer {index}\nEngineer\nBuilt services between {date_range}.\n"
            text += block
            blocks.append({"start": start, "end": len(text), "text": block})
            for quote, label in ((f"Employer {index}", "organization"), ("Engineer", "job title"), (date_range, "date")):
                if missing_ner and label == "date":
                    continue
                offset = text.index(quote, start)
                entities.append({"entity_id": f"mention-{len(entities)}", "status": "not_applicable",
                                 "entity": {"text": quote, "label": label, "start": offset, "end": offset + len(quote),
                                            "section": "experience", "source_field": "section_texts.experience"}})
        features = {"cv_id": "scope.pdf", "section_texts": {"experience": text, "education": education},
                    "experience_refined": blocks}
        original = copy.deepcopy(features)
        result = OntologyPopulator(as_of=REFERENCE).populate(features, {"entity_mappings": {"not_applicable": entities}})
        self.assertEqual(features, original)
        experiences = sorted((n for n in result.nodes if n.rdf_type.endswith("#Experience")),
                             key=lambda n: result.report["nodes"][n.id]["block"]["start"])
        return experiences, result, features

    def test_each_job_uses_its_own_range_with_no_education_year_fallback(self):
        jobs, result, features = self.populate(["September - Present", "June - August 2025", "May 2024 - Present"])
        self.assertEqual([j.years_experience for j in jobs], [[], [0.17], [2.5]])
        again, _, _ = self.populate(["September - Present", "June - August 2025", "May 2024 - Present"],
                                    education="Master, January 2023 - December 2026")
        self.assertEqual([j.years_experience for j in jobs], [j.years_experience for j in again])
        document = json.loads(OntologyDocument(cv_id="scope.pdf", nodes=result.nodes).model_dump_json())
        for node in document["nodes"]:
            duration = node["metadata"].get("duration")
            if duration is not None:
                self.assertEqual(duration["source_field"], "section_texts.experience")
                self.assertEqual(features["section_texts"]["experience"][duration["start"]:duration["end"]], duration["text"])
                self.assertEqual(duration["as_of_month"], "2026-11")

    def test_header_dates_do_not_depend_on_ner_detection(self):
        jobs, _, _ = self.populate(["January 2022 - January 2024"], missing_ner=True)
        self.assertEqual(jobs[0].years_experience, [2.0])

    def test_body_ranges_are_not_used_as_employment_dates(self):
        jobs, _, _ = self.populate(["January 2022 - January 2024"], body_range=True)
        self.assertEqual(jobs[0].years_experience, [])

    def test_overlapping_jobs_are_calculated_independently(self):
        jobs, _, _ = self.populate(["January 2022 - January 2024", "June 2022 - June 2024"])
        self.assertEqual([j.years_experience for j in jobs], [[2.0], [2.0]])
