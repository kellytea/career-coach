"""Command line for the matching module.

    python -m career_coach.matching.cli jobs --kaggle job_postings.csv --out jobs.jsonl --limit 5000
    python -m career_coach.matching.cli match --resumes resumes.json --jobs jobs.jsonl --system S7
    python -m career_coach.matching.cli match --resumes resumes.json --jobs outputs/job_postings --postings-csv data/postings.csv
    python -m career_coach.matching.cli pool --resumes resumes.json --jobs jobs.jsonl --out pool.csv
    python -m career_coach.matching.cli evaluate --resumes resumes.json --jobs jobs.jsonl --qrels pool.csv

``--embeddings hashing`` replaces both Sentence-BERT models with an offline hashing
encoder. Use it only to check the pipeline; its numbers are not S4-S7 results.
"""

import argparse
import json
import sys
from pathlib import Path

from .embeddings import MINILM, MPNET, HashingEmbedder, SentenceTransformerEmbedder
from .evaluation import evaluate, pool, write_csv
from .matcher import SYSTEMS, Matcher
from .profiles import load_any_jobs, load_jobs, load_kaggle_postings, load_resumes, save_jobs
from .skills import SkillExtractor
from .taxonomy import DEFAULT_TAXONOMY_DIR, Taxonomy


def build_matcher(args, jobs) -> Matcher:
    if args.embeddings == "hashing":
        minilm, mpnet = HashingEmbedder(name="hashing-a"), HashingEmbedder(dim=768, name="hashing-b")
    elif args.embeddings == "none":
        minilm = mpnet = None
    else:
        minilm, mpnet = SentenceTransformerEmbedder(args.minilm), SentenceTransformerEmbedder(args.mpnet)
    cache = None if args.embeddings == "hashing" else getattr(args, "cache_dir", None)
    return Matcher.build(jobs, minilm=minilm, mpnet=mpnet, tau=args.tau, depth=args.depth, cache_dir=cache)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    jobs_cmd = sub.add_parser("jobs", help="Extract skills from postings into a JSONL file")
    jobs_cmd.add_argument("--kaggle", type=Path, help="Kaggle LinkedIn job_postings.csv")
    jobs_cmd.add_argument("--jsonl", type=Path, help="JSONL with job_id, title, description")
    jobs_cmd.add_argument("--out", type=Path, required=True)
    jobs_cmd.add_argument("--limit", type=int)

    for name in ("match", "pool", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--resumes", type=Path, required=True, help="Parser output JSON")
        cmd.add_argument("--jobs", type=Path, required=True,
                         help="JSONL from the jobs command, or job_ingestion output (folder of graph JSON files)")
        cmd.add_argument("--postings-csv", type=Path,
                         help="Source postings.csv, to add description text to job_ingestion postings")
        cmd.add_argument("--embeddings", choices=("sbert", "hashing", "none"), default="sbert")
        cmd.add_argument("--minilm", default=MINILM)
        cmd.add_argument("--mpnet", default=MPNET)
        cmd.add_argument("--tau", type=float, default=0.7)
        cmd.add_argument("--depth", type=int, default=100)
        cmd.add_argument("--cache-dir", type=Path, default=Path(".cache/embeddings"),
                         help="Reuse posting embeddings between runs")
    sub.choices["match"].add_argument("--system", default="S7", choices=sorted(SYSTEMS))
    sub.choices["match"].add_argument("--cv-id", help="Only this resume")
    sub.choices["match"].add_argument("--top-k", type=int, default=10)
    sub.choices["pool"].add_argument("--systems", nargs="+", default=list(SYSTEMS))
    sub.choices["pool"].add_argument("--pool-depth", type=int, default=10)
    sub.choices["pool"].add_argument("--out", type=Path, required=True)
    sub.choices["evaluate"].add_argument("--qrels", type=Path, required=True)
    sub.choices["evaluate"].add_argument("--systems", nargs="+", default=list(SYSTEMS))
    sub.choices["evaluate"].add_argument("--baseline", default="S2")
    sub.choices["evaluate"].add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    taxonomy = Taxonomy(args.taxonomy)
    extractor = SkillExtractor(taxonomy)

    if args.command == "jobs":
        if bool(args.kaggle) == bool(args.jsonl):
            parser.error("Give exactly one of --kaggle or --jsonl")
        jobs = (load_kaggle_postings(args.kaggle, extractor, args.limit) if args.kaggle
                else load_jobs(args.jsonl, extractor)[:args.limit])
        save_jobs(jobs, args.out)
        without = sum(1 for j in jobs if not j.required and not j.preferred)
        print(f"Saved {len(jobs)} postings to {args.out} ({without} with no recognized O*NET skills)")
        return

    resumes = load_resumes(args.resumes, taxonomy)
    jobs = load_any_jobs(args.jobs, extractor, args.postings_csv)
    matcher = build_matcher(args, jobs)

    if args.command == "match":
        selected = [r for r in resumes if not args.cv_id or r.cv_id == args.cv_id]
        if not selected:
            parser.error(f"No resume with cv_id {args.cv_id}")
        output = {r.cv_id: [m.to_dict() for m in matcher.rank(r, args.system, args.top_k)] for r in selected}
        json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
        print()
    elif args.command == "pool":
        systems = [s for s in args.systems if s in matcher.available_systems()]
        rows = pool(matcher, resumes, systems, args.pool_depth)
        write_csv(rows, args.out)
        print(f"Wrote {len(rows)} resume-job pairs for grading from {systems} to {args.out}")
    else:
        systems = [s for s in args.systems if s in matcher.available_systems()]
        from .evaluation import load_qrels
        report = evaluate(matcher, resumes, load_qrels(args.qrels), systems,
                          args.baseline if args.baseline in systems else None)
        text = json.dumps({k: v for k, v in report.items() if k != "per_query"}, indent=2)
        print(text)
        if args.out:
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
