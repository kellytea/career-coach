"""Resume-to-job matching: classical baselines, dense retrieval, a knowledge-graph
skill score and Reciprocal Rank Fusion, with per-match explanations.

Inputs are the parser's ontology JSON (``resumes.json``/``outputs.json``) and job
postings. Skill and role IDs are the same ``urn:career-coach:concept:...`` IDs the
parser emits, so resume skills and job skills can be compared directly.
"""

from .profiles import (JobPosting, ResumeProfile, SkillRef, load_any_jobs, load_graph_jobs, load_jobs,
                       load_resumes, save_jobs)
from .matcher import SYSTEMS, Match, Matcher
from .taxonomy import Taxonomy

__all__ = ["JobPosting", "ResumeProfile", "SkillRef", "load_any_jobs", "load_graph_jobs", "load_jobs",
           "load_resumes", "save_jobs",
           "SYSTEMS", "Match", "Matcher", "Taxonomy"]
