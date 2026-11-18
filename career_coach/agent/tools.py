"""The agent's four tools: search_jobs, explain_match, skill_gap, draft_outreach.

Every tool works on the active resume and returns JSON-serializable data built
from the matcher's output, so the LLM only rephrases retrieved facts.
"""

from collections import Counter

from career_coach.matching.matcher import Matcher
from career_coach.matching.profiles import ResumeProfile
from career_coach.matching.text import tokenize

from .grounding import ClaimVerifier, build_context, format_facts
from .llm import ChatModel, LLMError, parse_json


GENERIC_QUERY_TERMS = frozenset({"job", "jobs", "role", "roles", "position", "positions", "work", "opening",
                                 "openings", "opportunity", "opportunities", "find", "show", "me", "good", "best"})

TOOL_SPECS = [
    {"name": "search_jobs", "args": {"query": "optional keywords such as 'data engineer remote'",
                                     "top_k": "number of postings, default 5"},
     "description": "Rank postings for the candidate; optional keywords filter the ranked list."},
    {"name": "explain_match", "args": {"job_id": "posting ID"},
     "description": "Why a posting matches: matched, similar and missing skills, ranker breakdown, evidence."},
    {"name": "skill_gap", "args": {"job_id": "optional posting ID; omit to summarize the top matches"},
     "description": "Missing skills for one posting, or the skills most often missing across top matches."},
    {"name": "draft_outreach", "args": {"job_id": "posting ID", "recipient": "'recruiter' or 'referral'",
                                        "notes": "optional extra instructions from the user"},
     "description": "Draft a short outreach message for the user to review; it is never sent."},
]

DRAFT_PROMPT = """You write short, honest job outreach messages for a candidate.
Use ONLY the facts below. Name a skill as the candidate's only if it appears in the candidate facts (R...).
Do not state a number of years of experience unless a candidate fact gives it. Do not invent employers,
projects, achievements or names. Use "[Your Name]" for the signature. Keep the message under 150 words.
Return JSON: {"subject": "...", "message": "..."}"""


class AssistantTools:
    def __init__(self, matcher: Matcher, resume: ResumeProfile, verifier: ClaimVerifier,
                 model: ChatModel | None = None, *, system: str = "S7", context_mode: str = "graph"):
        self.matcher, self.resume, self.verifier, self.model = matcher, resume, verifier, model
        self.system, self.context_mode = system, context_mode
        self.shortlist: list[str] = []     # job IDs from the latest search, for "the second one"
        self.seen_job_ids: set[str] = set()

    # -- dispatcher -------------------------------------------------------------------
    def call(self, name: str, args: dict | None) -> dict:
        tool = {"search_jobs": self.search_jobs, "explain_match": self.explain_match,
                "skill_gap": self.skill_gap, "draft_outreach": self.draft_outreach}.get(name)
        if tool is None:
            return {"error": f"Unknown tool {name!r}. Available: {[t['name'] for t in TOOL_SPECS]}"}
        allowed = {a for t in TOOL_SPECS if t["name"] == name for a in t["args"]}
        args = dict(args or {})
        unexpected = set(args) - allowed
        if unexpected:
            return {"error": f"Unexpected arguments for {name}: {sorted(unexpected)}"}
        try:
            return tool(**args)
        except (KeyError, ValueError, TypeError) as error:
            return {"error": str(error).strip("'\"")}

    def _job_id(self, job_id) -> str:
        job_id = str(job_id).strip()
        if job_id not in self.matcher.by_id:
            raise KeyError(f"Unknown job_id {job_id}. Search first and use an ID from the results.")
        self.seen_job_ids.add(job_id)
        return job_id

    # -- tools ------------------------------------------------------------------------
    def search_jobs(self, query: str = "", top_k: int = 5) -> dict:
        top_k = max(1, min(int(top_k), 20))
        terms = [t for t in tokenize(query or "") if t not in GENERIC_QUERY_TERMS]

        def where(job):
            text = set(tokenize(f"{job.title} {job.description} {job.location}"))
            return all(t in text for t in terms)

        matches = self.matcher.rank(self.resume, self.system, top_k, where if terms else None)
        self.shortlist = [m.job.job_id for m in matches]
        self.seen_job_ids.update(self.shortlist)
        results = []
        for m in matches:
            item = m.to_dict()
            results.append({k: item[k] for k in ("job_id", "title", "company", "rank", "kg_score",
                                                  "matched_skills", "missing_required", "url")})
        return {"system": self.system, "query_terms": terms, "results": results,
                "note": None if results else "No posting matched every keyword; try fewer keywords."}

    def explain_match(self, job_id: str) -> dict:
        match = self.matcher.explain(self.resume, self._job_id(job_id), self.system)
        facts = build_context(match, self.resume, self.context_mode)
        return {**match.to_dict(), "context": [{"id": f.id, "text": f.text} for f in facts]}

    def skill_gap(self, job_id: str | None = None, top_n: int = 10) -> dict:
        if job_id:
            match = self.matcher.explain(self.resume, self._job_id(job_id), self.system).to_dict()
            return {"job_id": match["job_id"], "title": match["title"],
                    "missing_required": match["missing_required"], "missing_preferred": match["missing_preferred"],
                    "similar_skills_you_have": match["near_miss_skills"], "matched_skills": match["matched_skills"]}
        matches = self.matcher.rank(self.resume, self.system, top_n)
        self.seen_job_ids.update(m.job.job_id for m in matches)
        counts, examples = Counter(), {}
        for m in matches:
            for skill in m.graph.missing_required:
                counts[skill.label] += 1
                examples.setdefault(skill.label, m.job.job_id)
        return {"based_on_top_matches": len(matches),
                "most_common_missing_required": [{"skill": s, "postings": c, "example_job_id": examples[s]}
                                                 for s, c in counts.most_common(10)],
                "note": "Counts come from the extracted O*NET skills of your top-ranked postings."}

    def draft_outreach(self, job_id: str, recipient: str = "recruiter", notes: str = "") -> dict:
        if recipient not in {"recruiter", "referral"}:
            raise ValueError("recipient must be 'recruiter' or 'referral'")
        match = self.matcher.explain(self.resume, self._job_id(job_id), self.system)
        facts = build_context(match, self.resume, "graph")
        allowed = self.resume.skill_ids
        draft, verification, source, problems = None, None, "template", []
        if self.model is not None:
            messages = [{"role": "system", "content": DRAFT_PROMPT},
                        {"role": "user", "content": f"Recipient: {recipient}\nUser notes: {notes or 'none'}\n"
                                                    f"Facts:\n{format_facts(facts)}"}]
            for _ in range(2):   # one generation plus one corrected retry
                try:
                    candidate = parse_json(self.model.complete(messages, temperature=0.3))
                    subject, message = str(candidate.get("subject", "")).strip(), str(candidate.get("message", "")).strip()
                except LLMError as error:
                    problems.append(str(error))
                    continue
                if not message:
                    problems.append("empty message")
                    continue
                verification = self.verifier.check(f"{subject}\n{message}", allowed,
                                                   max_years=self.resume.years_experience, check_years=True)
                if verification.ok:
                    draft, source = {"subject": subject, "message": message}, "llm"
                    break
                problems.append(verification.feedback())
                messages += [{"role": "assistant", "content": message},
                             {"role": "user", "content": f"Rewrite it: {verification.feedback()}. Remove those claims."}]
        if draft is None:
            draft = self._template(match, recipient)
            verification = self.verifier.check(f"{draft['subject']}\n{draft['message']}", allowed,
                                               max_years=self.resume.years_experience, check_years=True)
        return {"job_id": match.job.job_id, "recipient": recipient, **draft, "source": source,
                "verification": verification.to_dict(), "llm_problems": problems,
                "requires_human_review": True, "sent": False}

    def _template(self, match, recipient: str) -> dict:
        """Fallback draft assembled only from verified profile and posting facts."""
        job = match.job
        at = f" at {job.company}" if job.company else ""
        held = [s.label for s in match.graph.matched_required + match.graph.matched_preferred][:5]
        roles = list(dict.fromkeys(self.resume.titles + self.resume.roles))[:2]
        background = []
        if roles:
            background.append("experience in roles such as " + " and ".join(roles))
        if held:
            background.append("hands-on work with " + ", ".join(held))
        lines = ["Hello,", "", f"I came across the {job.title} position{at} and I am interested in it."]
        if background:
            lines.append(f"My background includes {'; '.join(background)}, which lines up with the posting.")
        lines += ["", "Would you be open to a short conversation about the role?" if recipient == "recruiter"
                  else "Would you be willing to refer me, or share what the team looks for in candidates?",
                  "", "Thank you,", "[Your Name]"]
        return {"subject": f"Interest in the {job.title} role{at}", "message": "\n".join(lines)}
