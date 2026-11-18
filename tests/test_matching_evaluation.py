"""Ranking metrics, significance testing, agreement and the pooling/qrels workflow."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

from matching_support import SAMPLE_JOBS, extractor, resume_document, write_resumes

from career_coach.matching.embeddings import HashingEmbedder
from career_coach.matching.evaluation import (evaluate, holm, load_qrels, ndcg_at_k, paired_bootstrap, pool,
                                              precision_at_k, reciprocal_rank, weighted_kappa, write_csv)
from career_coach.matching.matcher import Matcher
from career_coach.matching.profiles import load_jobs, load_resumes


class MetricTests(unittest.TestCase):
    def test_precision_mrr_ndcg(self):
        grades = {"a": 2, "b": 0, "c": 1}
        ranked = ["b", "a", "x", "c"]
        self.assertEqual(precision_at_k(ranked, grades, 2), 0.5)
        self.assertEqual(precision_at_k(ranked, grades, 4), 0.5)       # unjudged "x" counts as not relevant
        self.assertEqual(reciprocal_rank(ranked, grades), 0.5)
        dcg = 3 / math.log2(3) + 1 / math.log2(5)
        ideal = 3 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(ndcg_at_k(ranked, grades, 10), dcg / ideal)
        self.assertEqual(ndcg_at_k(["a", "c"], grades, 10), 1.0)
        self.assertEqual(ndcg_at_k(["a"], {"a": 0}, 10), 0.0)

    def test_bootstrap(self):
        same = paired_bootstrap([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], iterations=500)
        self.assertEqual(same["mean_diff"], 0.0)
        self.assertEqual(same["p_value"], 1.0)
        better = paired_bootstrap([0.1] * 20, [0.5] * 20, iterations=500)
        self.assertAlmostEqual(better["mean_diff"], 0.4)
        self.assertEqual(better["p_value"], 0.0)
        with self.assertRaises(ValueError):
            paired_bootstrap([1], [1, 2])

    def test_holm_is_step_down(self):
        result = holm({"S5": 0.01, "S6": 0.04, "S7": 0.03})
        self.assertAlmostEqual(result["S5"]["p_holm"], 0.03)
        self.assertAlmostEqual(result["S7"]["p_holm"], 0.06)
        self.assertAlmostEqual(result["S6"]["p_holm"], 0.06)
        self.assertEqual([result[s]["significant"] for s in ("S5", "S7", "S6")], [True, False, False])

    def test_weighted_kappa(self):
        self.assertEqual(weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2]), 1.0)
        # Known value: two raters, linear weights.
        kappa = weighted_kappa([0, 0, 1, 2], [0, 1, 1, 2])
        self.assertAlmostEqual(kappa, 1 - 0.125 / 0.4375)   # matches sklearn cohen_kappa_score(weights="linear")
        with self.assertRaises(ValueError):
            weighted_kappa([], [])


class WorkflowTests(unittest.TestCase):
    def test_pool_grade_and_evaluate(self):
        jobs = load_jobs(SAMPLE_JOBS, extractor())
        matcher = Matcher.build(jobs, minilm=HashingEmbedder(name="a"), mpnet=HashingEmbedder(dim=768, name="b"),
                                use_faiss=False)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            resumes = load_resumes(write_resumes(directory / "r.json", resume_document("one.pdf"),
                                                 resume_document("two.pdf", skills=("Tableau", "Microsoft Excel"))))
            rows = pool(matcher, resumes, ["S1", "S2", "S7"], depth=3)
            self.assertEqual(len(rows), len({(r["cv_id"], r["job_id"]) for r in rows}))   # no duplicates
            self.assertNotIn("system", rows[0])                                            # graders are blind
            relevant = {"syn-backend", "syn-fullstack", "syn-analyst"}
            for row in rows:
                row["grade"] = 2 if row["job_id"] in relevant else 0
            write_csv(rows, directory / "graded.csv")
            qrels = load_qrels(directory / "graded.csv")
            report = evaluate(matcher, resumes, qrels, ["S1", "S2", "S7"], baseline="S2", iterations=200)
            self.assertEqual(report["resumes"], 2)
            self.assertEqual(set(report["comparisons"]), {"S1", "S7"})
            for summary in report["summary"].values():
                self.assertTrue(0 <= summary["nDCG@10"] <= 1)

            with (directory / "bad.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["cv_id", "job_id", "grade"])
                writer.writeheader()
                writer.writerow({"cv_id": "one.pdf", "job_id": "syn-backend", "grade": "3"})
            with self.assertRaises(ValueError):
                load_qrels(directory / "bad.csv")


if __name__ == "__main__":
    unittest.main()
