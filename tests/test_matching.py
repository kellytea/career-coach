"""Matching module: taxonomy identity, skill extraction, rankers, KG score, RRF, systems S1-S7."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from matching_support import (LEGACY_OUTPUT, SAMPLE_JOBS, TableEmbedder, concept, extractor, resume_document,
                              taxonomy, write_resumes)

from career_coach.matching.embeddings import HashingEmbedder, VectorIndex
from career_coach.matching.fusion import reciprocal_rank_fusion
from career_coach.matching.graph import KGSkillScorer
from career_coach.matching.matcher import SYSTEMS, Matcher
from career_coach.matching.profiles import (JobPosting, ResumeProfile, SkillRef, load_jobs, load_resumes,
                                            minimum_degree, minimum_years, save_jobs)
from career_coach.matching.rankers import BM25Ranker, JaccardRanker, TfidfRanker
from career_coach.matching.text import chunk_words, job_sections, relevant_job_text


def ref(label):
    c = concept(label)
    return SkillRef(c.id, c.label)


def job(job_id, title, description="", required=(), preferred=()):
    return JobPosting(job_id, title, description, required=[ref(x) for x in required],
                      preferred=[ref(x) for x in preferred])


def resume(*skills, cv_id="r"):
    return ResumeProfile(cv_id, f"urn:test:{cv_id}", skills=[ref(s) for s in skills])


class TaxonomyIdentityTests(unittest.TestCase):
    def test_parser_concept_ids_are_reproduced(self):
        # Every Skill/Role ID in the parser's sample output maps back to an O*NET label.
        nodes = json.loads(LEGACY_OUTPUT.read_text())["resumes"][0]["nodes"]
        concepts = [n for n in nodes if n["type"].endswith(("#Skill", "#Role"))]
        self.assertTrue(concepts)
        for node in concepts:
            self.assertIsNotNone(taxonomy().label(node["id"]), node["id"])

    def test_legacy_output_without_metadata_loads_labels(self):
        profile = load_resumes(LEGACY_OUTPUT, taxonomy())[0]
        labels = {s.label for s in profile.skills}
        self.assertIn("Python", labels)
        self.assertIn("Amazon Web Services AWS software", labels)
        self.assertIn("Software Developers", profile.roles)
        self.assertIsNone(profile.years_experience)   # not stated, not zero
        self.assertIsNone(profile.degree_rank)        # degree node exists but is not linked to the Person


class ResumeLoadingTests(unittest.TestCase):
    def test_only_asserted_relationships_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_resumes(Path(directory) / "resumes.json", resume_document())
            profile = load_resumes(path)[0]
        labels = [s.label for s in profile.skills]
        self.assertEqual(labels, ["Python", "Docker", "Oracle Java", "React"])
        self.assertNotIn("Kubernetes", labels)        # node exists but no hasSkill edge
        self.assertEqual(profile.years_experience, 3.5)
        self.assertEqual(profile.degree_rank, 2)
        self.assertEqual(profile.titles, ["Software Engineer"])
        self.assertIn("Skills: Python, Docker", profile.text())

    def test_missing_referenced_skill_is_an_error(self):
        document = resume_document()
        document["nodes"] = [n for n in document["nodes"] if not n["type"].endswith("#Skill")]
        with tempfile.TemporaryDirectory() as directory:
            path = write_resumes(Path(directory) / "resumes.json", document)
            with self.assertRaises(ValueError):
                load_resumes(path)


class SkillExtractionTests(unittest.TestCase):
    def labels(self, concepts):
        return [c.label for c in concepts]

    def test_required_preferred_and_ignored_sections(self):
        text = ("About us:\nWe love Python.\n\nRequirements:\n- Docker and Kubernetes\n"
                "Preferred Qualifications:\n- GraphQL\n\nBenefits:\n- Free Tableau licence")
        skills = extractor().extract_job_skills(text)
        self.assertEqual(self.labels(skills.required), ["Docker", "Kubernetes"])
        self.assertEqual(self.labels(skills.preferred), ["GraphQL"])   # Python (about) and Tableau (benefits) ignored

    def test_preferred_cue_is_sentence_scoped(self):
        skills = extractor().extract_job_skills("SQL, Apache Spark and BigQuery. Master's degree preferred.")
        self.assertEqual(self.labels(skills.required), ["Structured query language SQL", "Apache Spark", "BigQuery"])
        self.assertEqual(skills.preferred, [])

    def test_required_wins_over_preferred(self):
        skills = extractor().extract_job_skills("Requirements: Python.\nPython 3.12 is a plus.")
        self.assertEqual(self.labels(skills.required), ["Python"])
        self.assertEqual(skills.preferred, [])

    def test_conservative_single_word_rules(self):
        found = lambda text: [m.concept.label for m in extractor().find(text)]
        self.assertEqual(found("We harvest data daily."), [])                  # common word, lower case
        self.assertEqual(found("Bachelor's degree in Computer Science"), [])   # generic O*NET skill
        self.assertEqual(found("- React, TypeScript and JavaScript"), ["React", "TypeScript", "JavaScript"])
        self.assertEqual(found("React to customer incidents quickly."), [])    # sentence-initial verb
        self.assertEqual(found("Go the extra mile. Own our Go To Market strategy."), [])
        self.assertEqual(found("Languages: Python, Go, SQL"), ["Python", "Go", "Structured query language SQL"])
        self.assertEqual(found("Americans with Disabilities Act"), [])          # not the ACT! CRM
        self.assertEqual(found("Strong writing skills"), ["Writing"])
        self.assertEqual(found("Experience with docker, git and bash."), ["Docker", "Git", "Bash"])
        self.assertEqual(found("Knowledge of C++, C# and Node.js."), ["C++", "C#", "Node.js"])
        self.assertEqual(found("AWS and Amazon Web Services AWS software"),
                         ["Amazon Web Services AWS software"] * 2)              # parser's curated alias


class TextTests(unittest.TestCase):
    def test_chunks_overlap_by_ten_percent(self):
        words = [f"w{i}" for i in range(450)]
        chunks = chunk_words(" ".join(words), 200, 0.1)
        self.assertEqual([len(c.split()) for c in chunks], [200, 200, 90])
        self.assertEqual(chunks[1].split()[0], "w180")       # step = 180 words
        self.assertEqual(chunk_words("", 200), [])
        with self.assertRaises(ValueError):
            chunk_words("x", 200, 1.0)

    def test_sections_and_relevant_text(self):
        text = "Great team.\nRequirements:\nPython\nBenefits:\nGym\nAbout Us:\nWe started in 2010"
        kinds = [s.kind for s in job_sections(text)]
        self.assertEqual(kinds, ["other", "requirements", "benefits", "about"])
        kept = relevant_job_text(text)
        self.assertIn("Python", kept)
        self.assertNotIn("Gym", kept)
        self.assertNotIn("2010", kept)

    def test_years_and_degree_parsing(self):
        self.assertEqual(minimum_years("3-5 years of professional software experience"), 3.0)
        self.assertEqual(minimum_years("5+ years experience; 2 years of Python experience"), 2.0)
        self.assertIsNone(minimum_years("Experience with Python"))
        self.assertEqual(minimum_degree("Bachelor's degree required, Master's preferred"), 2)
        self.assertIsNone(minimum_degree("Scrum Master experience"))
        self.assertEqual(minimum_degree("PhD in Physics"), 4)


class JobFileTests(unittest.TestCase):
    def test_sample_jobs_extract_and_round_trip(self):
        jobs = load_jobs(SAMPLE_JOBS, extractor())
        backend = next(j for j in jobs if j.job_id == "syn-backend")
        self.assertEqual([s.label for s in backend.required], ["Python", "Django", "Docker", "Kubernetes"])
        self.assertEqual([s.label for s in backend.preferred], ["Amazon Web Services AWS software", "GraphQL"])
        self.assertEqual((backend.min_years, backend.min_degree_rank), (3.0, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.jsonl"
            save_jobs(jobs, path)
            again = load_jobs(path)                     # skills already present: no extractor needed
        self.assertEqual([j.to_dict() for j in again], [j.to_dict() for j in jobs])

    def test_duplicate_job_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.jsonl"
            row = json.dumps({"job_id": "1", "title": "x", "description": "y", "required": [], "preferred": []})
            path.write_text(row + "\n" + row + "\n")
            with self.assertRaises(ValueError):
                load_jobs(path)


class RankerTests(unittest.TestCase):
    def setUp(self):
        self.jobs = [job("a", "Python Django developer", "Build Python APIs with Django", ["Python", "Django"]),
                     job("b", "Nurse", "Patient care in hospitals", []),
                     job("c", "Frontend", "React user interfaces", ["React"])]
        self.resume = resume("Python", "Django", "Docker")

    def test_lexical_rankers_prefer_overlap_and_drop_zero_scores(self):
        for ranker in (TfidfRanker(), BM25Ranker()):
            ranked = ranker.fit(self.jobs).rank(self.resume)
            self.assertEqual(ranked[0][0], 0, ranker.name)
            self.assertNotIn(1, [i for i, _ in ranked], ranker.name)

    def test_tfidf_is_a_cosine(self):
        ranked = TfidfRanker().fit(self.jobs).rank(self.resume)
        self.assertTrue(all(0 < s <= 1 + 1e-9 for _, s in ranked))

    def test_jaccard_value(self):
        ranked = JaccardRanker().fit(self.jobs).rank(self.resume)
        self.assertEqual(ranked, [(0, 2 / 3)])          # {Python, Django} / {Python, Django, Docker}

    def test_vector_index_exact_search(self):
        index = VectorIndex(HashingEmbedder().encode(["python django", "patient care", "react ui"]), use_faiss=False)
        scores, ids = index.search(HashingEmbedder().encode(["django python"]), 2)
        self.assertEqual(ids[0][0], 0)
        self.assertGreaterEqual(scores[0][0], scores[0][1])


class FusionTests(unittest.TestCase):
    def test_rrf_scores_and_ranks(self):
        fused = reciprocal_rank_fusion({"x": [1, 2, 3], "y": [3, 1]}, k=60)
        scores = {item: score for item, score, _ in fused}
        self.assertAlmostEqual(scores[1], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(scores[3], 1 / 63 + 1 / 61)
        self.assertAlmostEqual(scores[2], 1 / 62)
        self.assertEqual([i for i, _, _ in fused], [1, 3, 2])
        self.assertEqual(fused[0][2], {"x": 1, "y": 2})

    def test_ties_are_deterministic_and_duplicates_ignored(self):
        fused = reciprocal_rank_fusion({"x": [5, 5, 6], "y": [6, 5]})
        self.assertEqual(fused[0][2], {"x": 1, "y": 2})   # second 5 in x is ignored
        self.assertEqual([i for i, _, _ in reciprocal_rank_fusion({"x": [2], "y": [1]})], [1, 2])


class KGScoreTests(unittest.TestCase):
    def setUp(self):
        # Required: Python, Kubernetes; preferred: GraphQL. Candidate: Python, Docker.
        self.job = job("j", "Platform", required=["Python", "Kubernetes"], preferred=["GraphQL"])
        self.resume = resume("Python", "Docker")

    def test_exact_matches_only(self):
        evidence = KGSkillScorer().evidence(self.resume, self.job)
        self.assertAlmostEqual(evidence.score, (2 * 1) / (2 * 2 + 1))    # 0.4
        self.assertEqual([s.label for s in evidence.missing_required], ["Kubernetes"])
        self.assertEqual([s.label for s in evidence.missing_preferred], ["GraphQL"])

    def test_similar_skill_gets_partial_credit_at_tau(self):
        table = {"Kubernetes": [1, 0, 0, 0], "Docker": [0.8, 0.6, 0, 0], "GraphQL": [0, 0, 1, 0],
                 "Python": [0, 0, 0, 1]}
        scorer = KGSkillScorer(TableEmbedder(table), tau=0.8)          # cos(Kubernetes, Docker) = 0.8
        evidence = scorer.evidence(self.resume, self.job)
        self.assertAlmostEqual(evidence.score, (2 * (1 + 0.5)) / 5)       # 0.6
        self.assertEqual([(n.job_skill.label, n.resume_skill.label) for n in evidence.near_misses],
                         [("Kubernetes", "Docker")])
        self.assertEqual(evidence.missing_required, [])
        below = KGSkillScorer(TableEmbedder(table), tau=0.81).evidence(self.resume, self.job)
        self.assertAlmostEqual(below.score, 0.4)

    def test_no_job_skills_scores_zero(self):
        self.assertEqual(KGSkillScorer().evidence(self.resume, job("x", "Empty")).score, 0.0)

    def test_invalid_weights_rejected(self):
        with self.assertRaises(ValueError):
            KGSkillScorer(partial_credit=1.5)


class MatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = load_jobs(SAMPLE_JOBS, extractor())
        cls.matcher = Matcher.build(cls.jobs, minilm=HashingEmbedder(name="a"),
                                    mpnet=HashingEmbedder(dim=768, name="b"), use_faiss=False)
        with tempfile.TemporaryDirectory() as directory:
            path = write_resumes(Path(directory) / "r.json", resume_document())
            cls.resume = load_resumes(path)[0]

    def test_every_system_runs(self):
        self.assertEqual(self.matcher.available_systems(), list(SYSTEMS))
        for system in SYSTEMS:
            ids = self.matcher.ranked_ids(self.resume, system, 5)
            self.assertTrue(ids, system)
            self.assertEqual(len(ids), len(set(ids)), system)

    def test_full_system_ranks_software_jobs_first(self):
        top = [m.job.job_id for m in self.matcher.rank(self.resume, "S7", 2)]
        self.assertEqual(set(top), {"syn-backend", "syn-fullstack"})

    def test_explanation_contents_and_rrf_score(self):
        match = self.matcher.explain(self.resume, "syn-backend", "S7")
        data = match.to_dict()
        self.assertEqual(set(data["rankers"]), {"bm25", "dense:mpnet", "kg"})
        self.assertEqual(data["matched_skills"], ["Python", "Docker"])
        self.assertEqual(data["missing_required"], ["Django", "Kubernetes"])
        self.assertEqual(data["requirements"]["years_experience"]["status"], "meets")   # 3.5 >= 3
        self.assertEqual(data["requirements"]["degree_level"]["status"], "meets")
        self.assertIsNotNone(data["best_matching_text"])
        expected = sum(1 / (60 + r["rank"]) for r in data["rankers"].values() if r["rank"])
        self.assertAlmostEqual(match.score, expected, places=6)

    def test_explain_agrees_with_rank(self):
        top = self.matcher.rank(self.resume, "S7", 3)
        again = self.matcher.explain(self.resume, top[1].job.job_id, "S7")
        self.assertEqual(again.rank, 2)
        self.assertAlmostEqual(again.score, top[1].score)

    def test_filter_and_errors(self):
        remote = self.matcher.rank(self.resume, "S7", 10, where=lambda j: j.location == "Remote")
        self.assertTrue(remote and all(m.job.location == "Remote" for m in remote))
        with self.assertRaises(KeyError):
            self.matcher.rank(self.resume, "S9")
        with self.assertRaises(KeyError):
            self.matcher.explain(self.resume, "missing")
        lexical_only = Matcher.build(self.jobs)
        self.assertEqual(lexical_only.available_systems(), ["S1", "S2", "S3"])
        with self.assertRaises(ValueError):
            lexical_only.rank(self.resume, "S7")



class OptionalBackendWiringTests(unittest.TestCase):
    """sentence-transformers and FAISS are optional; check the wiring with stand-in modules."""

    def test_sentence_transformer_embedder_uses_normalized_embeddings(self):
        import sys
        import types
        from unittest.mock import patch

        import numpy as np

        from career_coach.matching.embeddings import SentenceTransformerEmbedder

        calls = {}

        class FakeModel:
            def __init__(self, name, device=None):
                calls["name"] = name

            def encode(self, texts, **kwargs):
                calls["kwargs"] = kwargs
                return np.array([[3.0, 4.0]] * len(texts))

        module = types.SimpleNamespace(SentenceTransformer=FakeModel)
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            vectors = SentenceTransformerEmbedder("some/model").encode(["a", "b"])
        self.assertEqual(calls["name"], "some/model")
        self.assertTrue(calls["kwargs"]["normalize_embeddings"])
        np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.6, 0.8]], rtol=1e-6)

    def test_faiss_backend_is_used_when_installed(self):
        import sys
        import types
        from unittest.mock import patch

        import numpy as np

        class FakeFlatIP:
            def __init__(self, dim):
                self.vectors = np.zeros((0, dim), np.float32)

            def add(self, vectors):
                self.vectors = np.vstack([self.vectors, vectors])

            def search(self, queries, k):
                sims = queries @ self.vectors.T
                ids = np.argsort(-sims, axis=1)[:, :k]
                return np.take_along_axis(sims, ids, axis=1), ids

        vectors = HashingEmbedder().encode(["python django", "patient care"])
        with patch.dict(sys.modules, {"faiss": types.SimpleNamespace(IndexFlatIP=FakeFlatIP)}):
            index = VectorIndex(vectors)
            self.assertEqual(index.backend, "faiss")
            _, ids = index.search(HashingEmbedder().encode(["django"]), 1)
        self.assertEqual(ids[0][0], 0)
        self.assertEqual(VectorIndex(vectors, use_faiss=False).backend, "numpy")


class EmbeddingCacheTests(unittest.TestCase):
    def test_posting_embeddings_are_reused(self):
        from career_coach.matching.rankers import DenseRanker

        class CountingEmbedder(HashingEmbedder):
            calls = 0

            def encode(self, texts):
                CountingEmbedder.calls += 1
                return super().encode(texts)

        jobs = [job("a", "Python developer", "Python APIs"), job("b", "Nurse", "Patient care")]
        with tempfile.TemporaryDirectory() as directory:
            first = DenseRanker(CountingEmbedder(), cache_dir=directory, use_faiss=False).fit(jobs)
            second = DenseRanker(CountingEmbedder(), cache_dir=directory, use_faiss=False).fit(jobs)
            changed = DenseRanker(CountingEmbedder(), cache_dir=directory, use_faiss=False).fit(jobs[:1])
        self.assertEqual(CountingEmbedder.calls, 2)            # second fit read the cache
        self.assertTrue((first.index.vectors == second.index.vectors).all())
        self.assertEqual(len(changed.index), 1)


class GraphJobTests(unittest.TestCase):
    """Postings exported by career_coach.job_ingestion (Neo4j-style graph JSON)."""

    FOLDER = Path(__file__).resolve().parent / "fixtures" / "graph_jobs"
    CSV = Path(__file__).resolve().parent / "fixtures" / "graph_postings.csv"

    def test_graph_export_is_read(self):
        from career_coach.matching.profiles import load_any_jobs, load_graph_jobs

        jobs = {j.job_id: j for j in load_graph_jobs(self.FOLDER)}
        backend = jobs["101"]
        self.assertEqual((backend.title, backend.company), ("Backend Engineer", "Acme"))
        self.assertEqual([s.label for s in backend.required], ["Python", "Docker", "CSS"])
        self.assertEqual(backend.required[0].id, concept("Python").id)        # same IDs as the resume parser
        self.assertEqual((backend.min_years, backend.min_degree_rank), (3.0, 2))
        self.assertEqual(backend.description, "")                            # the export carries no text
        self.assertEqual(jobs["102"].required, [])
        self.assertEqual(len(load_any_jobs(self.FOLDER)), 2)

    def test_descriptions_come_from_the_postings_csv(self):
        from career_coach.matching.profiles import load_graph_jobs

        jobs = {j.job_id: j for j in load_graph_jobs(self.FOLDER, self.CSV)}
        self.assertIn("Python services", jobs["101"].description)
        self.assertIn("Skills:\nCSS", jobs["101"].description)
        self.assertEqual((jobs["101"].location, jobs["101"].url), ("Phoenix, AZ", "https://example.com/101"))

    def test_unresolved_resume_skill_matches_unresolved_job_skill(self):
        from career_coach.matching.profiles import load_graph_jobs

        document = resume_document(skills=("Python",), unlinked_skills=())
        document["nodes"].append({"type": "http://example.org/jobmatch#Skill",
                                  "id": "urn:career-coach:resume:x:local-skill:abc", "metadata": {"label": "css"}})
        person = next(n for n in document["nodes"] if n["type"].endswith("#Person"))
        person["hasSkill"].append("urn:career-coach:resume:x:local-skill:abc")
        with tempfile.TemporaryDirectory() as directory:
            profile = load_resumes(write_resumes(Path(directory) / "r.json", document))[0]
        jobs = load_graph_jobs(self.FOLDER, self.CSV)
        evidence = KGSkillScorer().evidence(profile, next(j for j in jobs if j.job_id == "101"))
        self.assertEqual([s.label for s in evidence.matched_required], ["Python", "CSS"])
        self.assertEqual([s.label for s in evidence.missing_required], ["Docker"])
        matcher = Matcher.build(jobs, minilm=HashingEmbedder(name="a"), mpnet=HashingEmbedder(dim=768, name="b"),
                                use_faiss=False)
        self.assertEqual(matcher.rank(profile, "S7", 1)[0].job.job_id, "101")


if __name__ == "__main__":
    unittest.main()
