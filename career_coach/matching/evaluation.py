"""Ranking evaluation for the matching module.

Workflow: ``pool`` the top results of every system per resume into a blind CSV,
graders fill ``grade`` (0 not relevant, 1 partly relevant, 2 strong match),
then ``evaluate`` computes P@5, P@10, MRR and nDCG@10 per system and compares
systems with a paired bootstrap over resumes, Holm-corrected across comparisons.
"""

import csv
import math
import random
from pathlib import Path

import numpy as np


GRADES = (0, 1, 2)


def precision_at_k(ranked: list[str], grades: dict[str, int], k: int, threshold: int = 1) -> float:
    """Fraction of the top k that is relevant; unjudged items count as not relevant."""
    return sum(1 for d in ranked[:k] if grades.get(d, 0) >= threshold) / k


def reciprocal_rank(ranked: list[str], grades: dict[str, int], threshold: int = 1) -> float:
    return next((1.0 / i for i, d in enumerate(ranked, 1) if grades.get(d, 0) >= threshold), 0.0)


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float:
    """nDCG with gain 2^g - 1; the ideal ranking uses every judged item for the query."""
    def dcg(values):
        return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(values, 1))
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    return dcg([grades.get(d, 0) for d in ranked[:k]]) / ideal if ideal else 0.0


def paired_bootstrap(a: list[float], b: list[float], iterations: int = 10000, seed: int = 0) -> dict:
    """Mean difference (b - a) over paired queries, 95% CI and two-sided p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not len(a):
        raise ValueError("Paired bootstrap needs two equal-length, non-empty lists")
    diff = b - a
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (iterations, len(diff)))].mean(axis=1)
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci95": [float(low), float(high)], "p_value": float(p)}


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    result, m, running, rejecting = {}, len(ordered), 0.0, True
    for i, (name, p) in enumerate(ordered):
        adjusted = min(1.0, max(running, (m - i) * p))
        running = adjusted
        rejecting = rejecting and adjusted <= alpha
        result[name] = {"p_value": p, "p_holm": adjusted, "significant": rejecting}
    return result


def weighted_kappa(first: list[int], second: list[int], categories=GRADES, weights: str = "linear") -> float:
    """Cohen's weighted kappa for ordinal grades (linear or quadratic weights)."""
    if len(first) != len(second) or not first:
        raise ValueError("Kappa needs two equal-length, non-empty lists")
    index = {c: i for i, c in enumerate(categories)}
    n = len(categories)
    observed = np.zeros((n, n))
    for x, y in zip(first, second):
        observed[index[x], index[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    power = 1 if weights == "linear" else 2
    w = np.array([[abs(i - j) ** power for j in range(n)] for i in range(n)]) / (n - 1) ** power
    denominator = (w * expected).sum()
    return 1.0 - (w * observed).sum() / denominator if denominator else 1.0


# ---------------------------------------------------------------------------
# Pooling and qrels
# ---------------------------------------------------------------------------

def pool(matcher, resumes, systems, depth: int = 10, seed: int = 0) -> list[dict]:
    """Union of each system's top-``depth`` per resume, shuffled so graders cannot see the system."""
    rng = random.Random(seed)
    rows = []
    for resume in resumes:
        ids = list(dict.fromkeys(j for s in systems for j in matcher.ranked_ids(resume, s, depth)))
        rng.shuffle(ids)
        for job_id in ids:
            job = matcher.job(job_id)
            rows.append({"cv_id": resume.cv_id, "job_id": job_id, "title": job.title, "company": job.company,
                         "grade": "", "grader": "", "notes": ""})
    return rows


def write_csv(rows: list[dict], path) -> None:
    if not rows:
        raise ValueError("Nothing to write")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_qrels(path) -> dict[str, dict[str, int]]:
    """cv_id -> {job_id: grade}. Rows with an empty grade are skipped; grades must be 0, 1 or 2."""
    qrels: dict[str, dict[str, int]] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not str(row.get("grade", "")).strip():
                continue
            grade = int(row["grade"])
            if grade not in GRADES:
                raise ValueError(f"Grade must be 0, 1 or 2, got {grade} for {row['cv_id']}/{row['job_id']}")
            qrels.setdefault(row["cv_id"], {})[row["job_id"]] = grade
    return qrels


def evaluate(matcher, resumes, qrels, systems, baseline: str | None = None,
             depth: int = 10, iterations: int = 10000) -> dict:
    """Per-system means and per-resume values; optional bootstrap comparisons against ``baseline``."""
    judged = [r for r in resumes if r.cv_id in qrels]
    if not judged:
        raise ValueError("No resume in the qrels file matches the loaded resumes")
    per_query = {}
    for system in systems:
        rows = []
        for resume in judged:
            ranked, grades = matcher.ranked_ids(resume, system, depth), qrels[resume.cv_id]
            rows.append({"cv_id": resume.cv_id, "P@5": precision_at_k(ranked, grades, 5),
                         "P@10": precision_at_k(ranked, grades, 10), "MRR": reciprocal_rank(ranked, grades),
                         "nDCG@10": ndcg_at_k(ranked, grades, 10),
                         "unjudged@10": sum(1 for d in ranked[:10] if d not in grades)})
        per_query[system] = rows
    metrics = ("P@5", "P@10", "MRR", "nDCG@10")
    summary = {s: {m: float(np.mean([r[m] for r in rows])) for m in (*metrics, "unjudged@10")}
               for s, rows in per_query.items()}
    report = {"resumes": len(judged), "summary": summary, "per_query": per_query}
    if baseline:
        tests = {}
        for system in systems:
            if system != baseline:
                tests[system] = paired_bootstrap([r["nDCG@10"] for r in per_query[baseline]],
                                                 [r["nDCG@10"] for r in per_query[system]], iterations)
        corrected = holm({s: t["p_value"] for s, t in tests.items()})
        report["comparisons"] = {s: {**t, **corrected[s], "metric": "nDCG@10", "baseline": baseline}
                                 for s, t in tests.items()}
    return report
