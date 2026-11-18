"""Explicit extraction remains complete when NER is empty or disagrees."""

import re
import unittest
from types import SimpleNamespace

from career_coach.data_extraction.entity_extraction import EntityExtractor


class ExplicitEntityTests(unittest.TestCase):
    def features(self):
        education = "Computer Science, B.S.\n2020 - 2024\nComputer Science, M.S.\nExpected 2027"
        split = education.index("Computer Science, M.S.")
        return {"section_texts": {"skills": "Skills: C/C++, HTML/CSS, .NET, LangChain, AC/DC, https://example.com/a", "education": education},
                "education_refined": {"degree_entries": [
                    {"start": 0, "end": split, "text": education[:split]},
                    {"start": split, "end": len(education), "text": education[split:]}]}}

    def test_empty_ner_retains_explicit_skills_and_both_degrees(self):
        model = SimpleNamespace(config=SimpleNamespace(max_len=256, max_width=8),
                                data_processor=SimpleNamespace(words_splitter=lambda text: [(m[0], m.start(), m.end()) for m in re.finditer(r"\S+", text)]),
                                predict_entities=lambda *args, **kwargs: [])
        features = self.features()
        entities = EntityExtractor(model=model).extract_entities(features)
        self.assertEqual({e["text"] for e in entities if e["section"] == "skills"},
                         {"C", "C++", "HTML", "CSS", ".NET", "LangChain", "AC/DC"})
        self.assertEqual({e["text"] for e in entities if e["label"] == "degree"}, {"B.S.", "M.S."})
        for entity in entities:
            self.assertEqual(features["section_texts"][entity["section"]][entity["start"]:entity["end"]], entity["text"])
            self.assertEqual(entity["extraction_sources"], ["rules"])
            self.assertIsNone(entity["score"])

    def test_partial_degree_prediction_conflict_preserves_seed_span(self):
        seeds = EntityExtractor.explicit_entities(self.features())
        degree = next(e for e in seeds if e["text"] == "B.S.")
        prediction = {**degree, "text": "B.S", "end": degree["end"] - 1, "label": "academic major", "score": 0.8}
        merged = EntityExtractor.merge_entities(seeds, [prediction])
        matching = [e for e in merged if e["section"] == "education" and e["start"] == degree["start"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["label"], "degree")
        self.assertEqual(matching[0]["text"], "B.S.")
        self.assertEqual(matching[0]["classification_conflicts"], ["academic major"])
        self.assertEqual(set(matching[0]["extraction_sources"]), {"rules", "gliner"})

    def test_compatible_prediction_enriches_label_without_duplicate(self):
        seeds = EntityExtractor.explicit_entities(self.features())
        language = next(e for e in seeds if e["text"] == "C++")
        merged = EntityExtractor.merge_entities(seeds, [{**language, "label": "programming language", "score": 0.9}])
        matching = [e for e in merged if e["section"] == "skills" and e["start"] == language["start"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["label"], "programming language")
        self.assertNotIn("classification_conflicts", matching[0])

    def test_ner_aggregate_does_not_recombine_split_list_items(self):
        features = {"section_texts": {"skills": "HTML/CSS"}}
        seeds = EntityExtractor.explicit_entities(features)
        aggregate = {**seeds[0], "text": "HTML/CSS", "start": 0, "end": 8, "score": 0.7}
        self.assertEqual([e["text"] for e in EntityExtractor.merge_entities(seeds, [aggregate])], ["HTML", "CSS"])

    def test_coursework_is_not_promoted_to_degrees_or_skills(self):
        self.assertEqual(EntityExtractor.explicit_entities({
            "section_texts": {"education": "Courses: MS systems, databases"},
            "education_refined": {"degree_entries": [], "coursework": ["MS systems", "databases"]}}), [])
