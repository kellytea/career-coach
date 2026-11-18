"""Interactive chat over one parsed resume and a set of job postings.

    python -m career_coach.agent.chat --resumes resumes.json --jobs jobs.jsonl

Direct commands work without an LLM:
    /search [keywords]      /explain <job_id>      /gap [job_id]
    /draft <job_id> [recruiter|referral]           /quit
Any other text goes to the LLM agent (needs VOYAGER_API_KEY and VOYAGER_MODEL).
"""

import argparse
import json
from pathlib import Path

from career_coach.matching.cli import build_matcher
from career_coach.matching.embeddings import MINILM, MPNET
from career_coach.matching.matcher import SYSTEMS
from career_coach.matching.profiles import load_any_jobs, load_resumes
from career_coach.matching.skills import SkillExtractor
from career_coach.matching.taxonomy import DEFAULT_TAXONOMY_DIR, Taxonomy

from .agent import CareerAgent
from .grounding import ClaimVerifier
from .llm import LLMError, VoyagerChatModel
from .tools import AssistantTools


def run_command(tools: AssistantTools, line: str) -> dict | None:
    parts = line.split()
    command, rest = parts[0].lower(), parts[1:]
    if command == "/search":
        return tools.call("search_jobs", {"query": " ".join(rest)})
    if command == "/explain" and rest:
        return tools.call("explain_match", {"job_id": rest[0]})
    if command == "/gap":
        return tools.call("skill_gap", {"job_id": rest[0]} if rest else {})
    if command == "/draft" and rest:
        return tools.call("draft_outreach", {"job_id": rest[0], "recipient": rest[1] if len(rest) > 1 else "recruiter"})
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resumes", type=Path, required=True, help="Parser output JSON")
    parser.add_argument("--jobs", type=Path, required=True,
                        help="Job postings JSONL, or job_ingestion output (folder of graph JSON files)")
    parser.add_argument("--postings-csv", type=Path, help="Source postings.csv for job_ingestion postings")
    parser.add_argument("--cv-id", help="Resume to use when the file has several")
    parser.add_argument("--system", default="S7", choices=sorted(SYSTEMS))
    parser.add_argument("--context", default="graph", choices=("graph", "text"),
                        help="GraphRAG facts or text-only RAG (baseline)")
    parser.add_argument("--embeddings", choices=("sbert", "hashing", "none"), default="sbert")
    parser.add_argument("--minilm", default=MINILM)
    parser.add_argument("--mpnet", default=MPNET)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/embeddings"),
                        help="Reuse posting embeddings between runs")
    parser.add_argument("--no-llm", action="store_true", help="Commands only; drafts use the template")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_DIR)
    args = parser.parse_args(argv)

    taxonomy = Taxonomy(args.taxonomy)
    extractor = SkillExtractor(taxonomy)
    resumes = load_resumes(args.resumes, taxonomy)
    resume = next((r for r in resumes if r.cv_id == args.cv_id), None) if args.cv_id else resumes[0]
    if resume is None:
        parser.error(f"No resume with cv_id {args.cv_id}")
    matcher = build_matcher(args, load_any_jobs(args.jobs, extractor, args.postings_csv))
    if args.system not in matcher.available_systems():
        parser.error(f"{args.system} is unavailable with --embeddings {args.embeddings}; "
                     f"available: {matcher.available_systems()}")

    model = None
    if not args.no_llm:
        try:
            model = VoyagerChatModel()
        except LLMError as error:
            print(f"LLM disabled ({error}). Direct commands still work.")
    verifier = ClaimVerifier(extractor)
    tools = AssistantTools(matcher, resume, verifier, model, system=args.system, context_mode=args.context)
    agent = CareerAgent(tools, model, verifier) if model else None

    print(f"Resume {resume.cv_id}: {len(resume.skills)} skills; {len(matcher.jobs)} postings; system {args.system}.")
    print("Commands: /search [keywords], /explain <job_id>, /gap [job_id], /draft <job_id> [recruiter|referral], /quit")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"/quit", "/exit"}:
            break
        if line.startswith("/"):
            result = run_command(tools, line)
            print(json.dumps(result, indent=2, ensure_ascii=False) if result is not None else "Unknown command.")
            continue
        if agent is None:
            print("The LLM is not configured; use a /command.")
            continue
        reply = agent.ask(line)
        print(f"\nassistant> {reply.answer}")
        if not reply.verified:
            print(f"[check] {'; '.join(reply.warnings) or 'answer could not be fully verified'}")


if __name__ == "__main__":
    main()
