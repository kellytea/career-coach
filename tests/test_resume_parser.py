"""Regression checks for the standalone rule-based parser's inline labels."""

import importlib
import re
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

ResumeParser = importlib.import_module("career_coach.data_extraction.parser").ResumeParser
EntityExtractor = importlib.import_module("career_coach.data_extraction.entity_extraction").EntityExtractor


class ResumeParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = ResumeParser()
        self.extractor = EntityExtractor()

    def prepare_sections(self, sections):
        texts = sections.get("section_texts", sections)
        sections["education_refined"] = self.parser.refine_education({"education": texts.get("education", "")})
        sections["experience_refined"] = self.parser.refine_experience(sections)
        return sections

    def make_model(self):
        model = Mock()
        model.config = SimpleNamespace(max_len=64, max_width=12)
        model.data_processor.words_splitter = lambda text: ((m[0], m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text))
        model.predict_entities.side_effect = lambda text, **kwargs: [{"text": text, "label": kwargs["labels"][0], "start": 0, "end": len(text), "score": 0.9}]
        return model

    def test_entity_extraction_keeps_each_resume_and_reuses_model(self):
        model = self.make_model()
        extractor = self.extractor
        with tempfile.TemporaryDirectory() as folder:
            for name in ("first", "second"):
                (Path(folder) / f"{name}.pdf").touch()
            with patch("gliner.GLiNER.from_pretrained", return_value=model) as load:
                with patch.object(ResumeParser, "pdf_to_text", side_effect=lambda path: f"Jane Doe\nEducation\n{path.stem} university"):
                    result = self.parser.run(folder)
                records = result.to_dict(orient="records")
                for row in records:
                    row["entities"] = extractor.extract_entities(row)
        self.assertEqual(len(result), 2)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(model.predict_entities.call_count, 4)
        for row in records:
            entity = next(e for e in row["entities"] if e["section"] == "education")
            self.assertEqual(entity["text"], row["section_texts"]["education"])
            self.assertEqual(entity["category"], "context")

    def test_refine_education_separates_degrees_courses_and_awards(self):
        sections = self.parser.extract_features(
            "Education\nArizona State University\nComputer Science, B.S.\nJanuary 2023 - May 2026\n"
            "Awards & Scholarships: New American University - President Award Scholar,\nDean's List\n"
            "Relevant Coursework: Discrete Math Structures,\nOperating Systems, Cloud Computing\n"
            "Computer Science, M.S.\nAugust 2025 - May 2027\n4+1 Accelerated Masters Program\n"
            "Experience\nDeveloper"
        )
        self.assertIn("\n", sections["education"])
        refined = self.parser.refine_education(sections)
        self.assertEqual(len(refined["degree_entries"]), 2)
        self.assertEqual(len(refined["context"]), 1)
        self.assertIn("january 2023", refined["degree_entries"][0]["text"])
        self.assertIn("august 2025", refined["degree_entries"][1]["text"])
        self.assertEqual(refined["coursework"], ["discrete math structures", "operating systems", "cloud computing"])
        self.assertEqual(refined["awards"], ["new american university - president award scholar", "dean's list"])
        self.assertNotIn("new american university", refined["degree_entries"][0]["text"])
        for blocks in self.parser._education_blocks(sections).values():
            for block in blocks:
                self.assertEqual(sections["education"][block["start"]:block["end"]], block["text"])

    def test_refined_items_join_wrapped_lines_and_skip_empty_items(self):
        sections = {"education": "Courses: Operating Systems, Principles of\nProgramming Languages,, Cloud Computing,\nAwards: President Award\nScholar, Dean's List,"}
        refined = self.parser.refine_education(sections)
        self.assertEqual(refined["coursework"], ["Operating Systems", "Principles of Programming Languages", "Cloud Computing"])
        self.assertEqual(refined["awards"], ["President Award Scholar", "Dean's List"])
        for blocks in self.parser._education_blocks(sections).values():
            for block in blocks:
                self.assertEqual(sections["education"][block["start"]:block["end"]], block["text"])

    def test_empty_education_refinement(self):
        self.assertTrue(all(not blocks for blocks in self.parser.refine_education({}).values()))

    def test_education_block_labels_and_offsets(self):
        sections = {"education": "Example University\nComputer Science, B.S.\nRelevant Coursework: Operating Systems\nAwards: Example Scholarship"}
        model = self.make_model()
        sections["education_refined"] = self.parser.refine_education(sections)
        self.extractor._entity_model = model
        entities = self.extractor.extract_entities(self.prepare_sections(sections))
        for entity in entities:
            self.assertEqual(sections["education"][entity["start"]:entity["end"]], entity["text"])
        self.assertEqual({e["category"] for e in entities}, {"context", "degree_entries"})
        for call in model.predict_entities.call_args_list:
            self.assertNotIn("Operating Systems", call.args[0])
            self.assertNotIn("Example Scholarship", call.args[0])
        refined = self.parser.refine_education(sections)
        self.assertEqual(refined["coursework"], ["Operating Systems"])
        self.assertEqual(refined["awards"], ["Example Scholarship"])

    def test_experience_blocks_preserve_each_job_and_raw_text(self):
        sections = self.parser.extract_features(
            "Jane Doe\nExperience\nExample Corp\nJanuary 2020 – June 2022\nSoftware Engineer | Python\n"
            "Built reliable APIs.\nAnother Corp\nJuly 2022 – Present\nSenior Engineer\nLed a team.\n"
            "Projects\nExample App | C++\nBuilt a useful tool."
        )
        blocks = self.parser.refine_experience(sections)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(blocks[0]["text"].startswith("Example Corp"))
        self.assertTrue(blocks[1]["text"].startswith("Another Corp"))
        self.assertIn("–", blocks[0]["text"])
        for block in blocks:
            self.assertEqual(sections["section_texts"]["experience"][block["start"]:block["end"]], block["text"])

    def test_section_labels_calls_and_source_offsets(self):
        sections = self.parser.extract_features(
            "Jane Doe\nProfile: Builds reliable systems.\nExperience\nExample Corp\n2020 - 2023\nEngineer\n"
            "Built useful APIs.\nProjects\nSample App | Python\nSkills: C++, SQL"
        )
        model = self.make_model()
        self.extractor._entity_model = model
        entities = self.extractor.extract_entities(self.prepare_sections(sections))
        self.assertEqual({e["section"] for e in entities}, {"title", "summary", "experience", "projects", "skills"})
        for entity in entities:
            source = sections["section_texts"][entity["section"]]
            self.assertEqual(source[entity["start"]:entity["end"]], entity["text"])
        calls = {tuple(call.kwargs["labels"]) for call in model.predict_entities.call_args_list}
        self.assertIn(tuple(EntityExtractor.SECTION_LABELS["experience"]), calls)
        self.assertIn(tuple(EntityExtractor.SECTION_LABELS["skills"]), calls)

    def test_long_blocks_are_covered_and_overlap_entities_deduplicated(self):
        model = self.make_model()
        model.config.max_len = 12
        model.predict_entities.side_effect = lambda text, **kwargs: ([{
            "text": "Python", "label": "technical skill", "score": 0.9,
            "start": text.index("Python"), "end": text.index("Python") + 6,
        }] if "Python" in text else [])
        self.extractor._entity_model = model
        text = "word " * 30 + "Python"
        entities = self.extractor.extract_entities({"summary": text})
        self.assertGreater(model.predict_entities.call_count, 1)
        self.assertEqual(len(entities), 1)
        self.assertEqual(text[entities[0]["start"]:entities[0]["end"]], "Python")
        for call in model.predict_entities.call_args_list:
            self.assertLessEqual(len(list(model.data_processor.words_splitter(call.args[0]))), 12)

    def test_unlabeled_summary_with_wrapped_sentences(self):
        result = self.parser.extract_features(
            "Jane Doe\nSoftware Engineer\njane@example.com | (602) 555-0100\n\n"
            "Software engineer with experience in\n"
            "building reliable services and accessible web applications.\n"
            "Enjoys working with multidisciplinary teams.\n"
            "Experience\nExample Corp\nSkills: Python, SQL"
        )
        self.assertEqual(result["title"], "jane doe software engineer jane@example.com | (602) 555-0100")
        self.assertEqual(result["summary"], "software engineer with experience in building reliable services and accessible web applications. enjoys working with multidisciplinary teams.")
        self.assertEqual(result["experience"], "example corp")
        self.assertEqual(result["skills"], "python, sql")

    def test_header_only_does_not_invent_summary(self):
        result = self.parser.extract_features(
            "Jane Doe\nSenior Software Engineer\nhttps://linkedin.com/in/jane\n"
            "jane@example.com | (602) 555-0100\nPhoenix, Arizona\nEducation\nExample University"
        )
        self.assertEqual(result["summary"], "")
        self.assertEqual(result["education"], "example university")

    def test_unlabeled_summary_without_sentence_punctuation(self):
        result = self.parser.extract_features(
            "Jane Doe\nExperienced software engineer building scalable services and reliable data pipelines\n"
            "Skills: Python"
        )
        self.assertEqual(result["title"], "jane doe")
        self.assertTrue(result["summary"].startswith("experienced software engineer"))

    def test_explicit_summary_remains_supported(self):
        result = self.parser.extract_features("Jane Doe\nProfile: Builds reliable systems.\nExperience\nDeveloper")
        self.assertEqual(result["summary"], "builds reliable systems.")
        self.assertEqual(result["experience"], "developer")

    def test_summary_inference_stops_at_first_main_section(self):
        prose = "Developed reliable software services and data pipelines for large multidisciplinary teams."
        for heading in ("Experience", "Education", "Projects", "Skills"):
            with self.subTest(heading=heading):
                result = self.parser.extract_features(f"Jane Doe\n{heading}\n{prose}")
                self.assertEqual(result["summary"], "")
                self.assertIn(prose.lower(), result[heading.lower()])

    def test_later_title_header_cannot_restart_summary_inference(self):
        result = self.parser.extract_features(
            "Jane Doe\nExperience\nExample Corp\nTitle: Software Engineer\n"
            "Developed reliable software services and data pipelines for large multidisciplinary teams."
        )
        self.assertEqual(result["summary"], "")

    def test_long_skill_labels_preserve_values(self):
        result = self.parser.extract_features(
            "TECHNICAL SKILLS\n"
            "Technical Skills: Java, JavaScript, TypeScript, C/C++, C#, Python, HTML/CSS, SQL\n"
            "Frameworks/Tools: React.js, Vue.js, Node.js, Django, Figma, .NET, LangChain, Git, Docker, Kubernetes, Jira, GCP, AWS"
        )
        skills = [value.strip() for value in result["skills"].split(",")]
        self.assertEqual(len(skills), 21)
        self.assertEqual(skills[0], "java")
        self.assertEqual(skills[8], "react.js")
        self.assertTrue({"c/c++", "c#", "html/css", ".net"}.issubset(skills))

    def test_short_inline_headers_do_not_discard_content(self):
        result = self.parser.extract_features("Skills: Python\nTools: Git\nProjects: Demo")
        self.assertEqual(result["skills"], "python, git")
        self.assertEqual(result["projects"], "demo")

    def test_subheadings_are_scoped_and_unknown_colons_are_preserved(self):
        result = self.parser.extract_features(
            "Summary\nTools: built internal tools\nSkills\n"
            "Frameworks / Tools: .NET, React.js\nAWS::Lambda\nExperience\nDeveloper"
        )
        self.assertEqual(result["summary"], "tools: built internal tools")
        self.assertEqual(result["skills"], ".net, react.js, aws::lambda")
        self.assertEqual(result["experience"], "developer")


if __name__ == "__main__":
    unittest.main()
