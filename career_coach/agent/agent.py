"""Single orchestrating LLM that calls one tool at a time and answers from tool results.

Protocol (JSON only, so it works on any OpenAI-compatible endpoint):
    {"action": "tool", "tool": "<name>", "args": {...}}
    {"action": "final", "answer": "<text>"}

Final answers are checked: every O*NET skill they name must belong to the candidate
or to a posting retrieved in this turn. One corrected retry is allowed; if the
answer still fails, it is returned with ``verified=False`` and the problems listed.
"""

import json
from dataclasses import dataclass, field

from .grounding import ClaimVerifier, allowed_for_answer, resume_facts, format_facts
from .llm import ChatModel, LLMError, parse_json
from .tools import TOOL_SPECS, AssistantTools


SYSTEM_PROMPT = """You are a career assistant helping one job seeker with job matching, skill gaps and outreach.
You can call these tools, one per reply:
{tools}

Reply with JSON only, either
{{"action": "tool", "tool": "<tool name>", "args": {{...}}}}
or, when you can answer,
{{"action": "final", "answer": "<answer for the user>"}}

Rules:
- Use tool results for every statement about postings, matches or skills. Do not invent postings, skills,
  employers or experience. If the tools do not give the information, say so.
- Refer to postings by job_id and title. When explaining a match, say which skills match, which are similar
  and which are missing.
- Outreach drafts come from draft_outreach. Show the draft and remind the user to review and edit it;
  nothing is ever sent automatically.
- Keep answers short and concrete.

Candidate profile (from the parsed resume):
{profile}"""

MAX_RESULT_CHARS = 6000


@dataclass
class AgentReply:
    answer: str
    verified: bool
    tool_calls: list[dict] = field(default_factory=list)
    verification: dict | None = None
    warnings: list[str] = field(default_factory=list)


class CareerAgent:
    def __init__(self, tools: AssistantTools, model: ChatModel, verifier: ClaimVerifier, *,
                 max_steps: int = 5, history_turns: int = 6):
        self.tools, self.model, self.verifier = tools, model, verifier
        self.max_steps, self.history_turns = max_steps, history_turns
        self.history: list[dict] = []
        tool_lines = "\n".join(f"- {t['name']}({', '.join(f'{k}: {v}' for k, v in t['args'].items())}): "
                               f"{t['description']}" for t in TOOL_SPECS)
        self.system_prompt = SYSTEM_PROMPT.format(tools=tool_lines,
                                                  profile=format_facts(resume_facts(tools.resume)) or "(empty)")

    def _state(self) -> str:
        if not self.tools.shortlist:
            return "No search has been run yet."
        return "Latest search results, in order: " + ", ".join(self.tools.shortlist)

    def ask(self, message: str) -> AgentReply:
        messages = [{"role": "system", "content": self.system_prompt},
                    *self.history[-2 * self.history_turns:],
                    {"role": "user", "content": f"{message}\n\n(State: {self._state()})"}]
        calls, warnings, retried_answer = [], [], False
        turn_jobs: set[str] = set()
        for _ in range(self.max_steps):
            try:
                raw = self.model.complete(messages)
                decision = parse_json(raw)
            except LLMError as error:
                warnings.append(str(error))
                messages.append({"role": "user", "content": "Reply with one valid JSON object only."})
                continue
            messages.append({"role": "assistant", "content": json.dumps(decision)})
            action = decision.get("action")
            if action == "tool":
                name, args = decision.get("tool", ""), decision.get("args") or {}
                before = set(self.tools.seen_job_ids)
                result = self.tools.call(name, args if isinstance(args, dict) else {})
                turn_jobs |= self.tools.seen_job_ids - before
                turn_jobs |= {r["job_id"] for r in result.get("results", [])}
                if result.get("job_id"):
                    turn_jobs.add(result["job_id"])
                calls.append({"tool": name, "args": args, "result": result})
                text = json.dumps(result, ensure_ascii=False)
                if len(text) > MAX_RESULT_CHARS:
                    text = text[:MAX_RESULT_CHARS] + " ...(truncated)"
                messages.append({"role": "user", "content": f"TOOL_RESULT {name}: {text}"})
                continue
            if action == "final" and isinstance(decision.get("answer"), str) and decision["answer"].strip():
                answer = decision["answer"].strip()
                allowed = allowed_for_answer(self.tools.matcher, self.tools.resume, turn_jobs)
                check = self.verifier.check(answer, allowed)
                if check.ok or retried_answer:
                    if not check.ok:
                        warnings.append("Answer still names unsupported skills: " + ", ".join(check.unsupported_skills))
                    self._remember(message, answer)
                    return AgentReply(answer, check.ok, calls, check.to_dict(), warnings)
                retried_answer = True
                messages.append({"role": "user", "content": f"Revise your answer: {check.feedback()}. "
                                                            "Only use skills from the tool results or the profile."})
                continue
            messages.append({"role": "user", "content": "Use action 'tool' or 'final' as specified."})
        answer = "I could not finish that request. Try rephrasing it or use a direct command such as /search."
        warnings.append("Step limit reached without a final answer")
        return AgentReply(answer, False, calls, None, warnings)

    def _remember(self, message: str, answer: str) -> None:
        # Stored in the same JSON form the model is asked to produce.
        self.history += [{"role": "user", "content": message},
                         {"role": "assistant", "content": json.dumps({"action": "final", "answer": answer})}]
