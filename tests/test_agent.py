"""Conversational agent: JSON protocol, tools, grounding context and claim verification. No network."""

import json
import tempfile
import unittest
from pathlib import Path

from matching_support import SAMPLE_JOBS, extractor, resume_document, write_resumes

from career_coach.agent.agent import CareerAgent
from career_coach.agent.grounding import ClaimVerifier, build_context, unsupported_skill_rate
from career_coach.agent.llm import LLMError, parse_json
from career_coach.agent.tools import AssistantTools
from career_coach.matching.embeddings import HashingEmbedder
from career_coach.matching.matcher import Matcher
from career_coach.matching.profiles import load_jobs, load_resumes


class ScriptedModel:
    """Returns prepared replies in order and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, *, temperature=0.0):
        self.calls.append([dict(m) for m in messages])   # snapshot: the agent keeps appending
        if not self.replies:
            raise AssertionError("Model called more times than expected")
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


def tool(name, **args):
    return {"action": "tool", "tool": name, "args": args}


def final(answer):
    return {"action": "final", "answer": answer}


class AgentTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = load_jobs(SAMPLE_JOBS, extractor())
        cls.matcher = Matcher.build(cls.jobs, minilm=HashingEmbedder(name="a"),
                                    mpnet=HashingEmbedder(dim=768, name="b"), use_faiss=False)
        with tempfile.TemporaryDirectory() as directory:
            cls.resume = load_resumes(write_resumes(Path(directory) / "r.json", resume_document()))[0]
        cls.verifier = ClaimVerifier(extractor())

    def tools(self, model=None, **kwargs):
        return AssistantTools(self.matcher, self.resume, self.verifier, model, **kwargs)


class ParseTests(unittest.TestCase):
    def test_fenced_and_wrapped_json(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json('Sure! {"a": {"b": 2}} done'), {"a": {"b": 2}})
        for bad in ("no json", "{broken", "[1, 2]"):
            with self.assertRaises(LLMError):
                parse_json(bad)


class VerifierTests(AgentTestCase):
    def test_skill_and_years_claims(self):
        allowed = self.resume.skill_ids
        ok = self.verifier.check("I build services in Python and Docker, and some Java.", allowed)
        self.assertTrue(ok.ok, ok.to_dict())
        self.assertEqual(ok.mentioned_skills, ["Python", "Docker", "Oracle Java"])
        bad = self.verifier.check("I am an expert in Kubernetes and Tableau.", allowed)
        self.assertEqual(bad.unsupported_skills, ["Kubernetes", "Tableau"])
        years = self.verifier.check("I have 5 years of experience.", allowed, max_years=3.5, check_years=True)
        self.assertEqual(years.unsupported_years, ["5 years"])
        unknown = self.verifier.check("2 years in industry", allowed, max_years=None, check_years=True)
        self.assertFalse(unknown.ok)                          # no stated years on the resume -> unsupported
        self.assertAlmostEqual(unsupported_skill_rate([ok, bad]), 2 / 5)


class ContextTests(AgentTestCase):
    def test_graph_and_text_modes(self):
        match = self.matcher.explain(self.resume, "syn-backend")
        graph = build_context(match, self.resume, "graph")
        self.assertTrue(any(f.id.startswith("G") and "hasSkill-> Python" in f.text for f in graph))
        self.assertTrue(any(f.id.startswith("R") for f in graph))
        text = build_context(match, self.resume, "text")
        self.assertTrue(text and all(f.id.startswith("T") for f in text))
        self.assertFalse(any("hasSkill" in f.text for f in text))      # baseline has no graph structure
        with self.assertRaises(ValueError):
            build_context(match, self.resume, "both")


class ToolTests(AgentTestCase):
    def test_search_with_keywords_and_shortlist(self):
        tools = self.tools()
        result = tools.call("search_jobs", {"query": "remote jobs", "top_k": 10})
        self.assertEqual(result["query_terms"], ["remote"])
        self.assertTrue(result["results"])
        self.assertTrue(all(self.matcher.job(r["job_id"]).location == "Remote" for r in result["results"]))
        self.assertEqual(tools.shortlist, [r["job_id"] for r in result["results"]])
        self.assertEqual(tools.call("search_jobs", {"query": "astronaut"})["results"], [])

    def test_errors_are_returned_not_raised(self):
        tools = self.tools()
        self.assertIn("error", tools.call("explain_match", {"job_id": "nope"}))
        self.assertIn("error", tools.call("explain_match", {"job_id": "syn-backend", "extra": 1}))
        self.assertIn("error", tools.call("delete_everything", {}))
        self.assertIn("error", tools.call("draft_outreach", {"job_id": "syn-backend", "recipient": "ceo"}))

    def test_skill_gap(self):
        tools = self.tools()
        one = tools.call("skill_gap", {"job_id": "syn-backend"})
        self.assertEqual(one["missing_required"], ["Django", "Kubernetes"])
        overall = tools.call("skill_gap", {})
        self.assertTrue(overall["most_common_missing_required"])

    def test_template_draft_without_model(self):
        draft = self.tools().call("draft_outreach", {"job_id": "syn-backend"})
        self.assertEqual(draft["source"], "template")
        self.assertTrue(draft["verification"]["ok"])
        self.assertTrue(draft["requires_human_review"])
        self.assertFalse(draft["sent"])
        self.assertIn("Python", draft["message"])
        self.assertNotIn("Django", draft["message"])                  # missing skills are not claimed

    def test_model_draft_is_corrected_then_accepted(self):
        model = ScriptedModel({"subject": "Hi", "message": "I have deep Kubernetes expertise."},
                              {"subject": "Backend role", "message": "I work with Python and Docker daily."})
        draft = self.tools(model).call("draft_outreach", {"job_id": "syn-backend"})
        self.assertEqual(draft["source"], "llm")
        self.assertEqual(draft["message"], "I work with Python and Docker daily.")
        self.assertIn("Kubernetes", draft["llm_problems"][0])
        self.assertIn("Kubernetes", model.calls[1][-1]["content"])    # the retry names the problem

    def test_model_draft_falls_back_to_template(self):
        model = ScriptedModel({"subject": "x", "message": "10 years of Kubernetes."},
                              {"subject": "x", "message": "Expert in Tableau."})
        draft = self.tools(model).call("draft_outreach", {"job_id": "syn-backend", "recipient": "referral"})
        self.assertEqual(draft["source"], "template")
        self.assertTrue(draft["verification"]["ok"])
        self.assertIn("refer me", draft["message"])
        self.assertEqual(len(draft["llm_problems"]), 2)


class AgentLoopTests(AgentTestCase):
    def agent(self, *replies):
        model = ScriptedModel(*replies)
        tools = self.tools(model)
        return CareerAgent(tools, model, self.verifier), model

    def test_tool_then_grounded_answer(self):
        agent, model = self.agent(tool("search_jobs", top_k=3),
                                  final("syn-backend fits: you have Python and Docker; Django is missing."))
        reply = agent.ask("Which jobs fit me?")
        self.assertTrue(reply.verified, reply.verification)
        self.assertEqual([c["tool"] for c in reply.tool_calls], ["search_jobs"])
        self.assertTrue(model.calls[1][-1]["content"].startswith("TOOL_RESULT search_jobs"))
        self.assertIn("Candidate skills: Python", model.calls[0][0]["content"])   # profile in system prompt
        self.assertEqual(len(agent.history), 2)

    def test_unsupported_answer_is_revised(self):
        agent, model = self.agent(final("You are strong in Tableau."), final("You are strong in Python."))
        reply = agent.ask("What am I good at?")
        self.assertTrue(reply.verified)
        self.assertEqual(reply.answer, "You are strong in Python.")
        self.assertIn("Tableau", model.calls[1][-1]["content"])

    def test_answer_still_unsupported_is_flagged(self):
        agent, _ = self.agent(final("Tableau expert."), final("Still a Tableau expert."))
        reply = agent.ask("What am I good at?")
        self.assertFalse(reply.verified)
        self.assertTrue(any("Tableau" in w for w in reply.warnings))

    def test_invalid_json_then_recovery_and_step_limit(self):
        agent, _ = self.agent("not json", final("Hello."))
        self.assertEqual(agent.ask("hi").answer, "Hello.")
        stuck, _ = self.agent(*[tool("skill_gap")] * 5)
        reply = stuck.ask("loop forever")
        self.assertFalse(reply.verified)
        self.assertEqual(len(reply.tool_calls), 5)

    def test_job_skills_from_tools_are_allowed_in_answers(self):
        # Kubernetes is not the candidate's, but the explained posting requires it.
        agent, _ = self.agent(tool("explain_match", job_id="syn-backend"),
                              final("syn-backend requires Kubernetes, which you do not list."))
        self.assertTrue(agent.ask("Explain syn-backend").verified)


if __name__ == "__main__":
    unittest.main()
