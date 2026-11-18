"""Deterministic O*NET skill mentions in free text (job postings, generated drafts).

The parser resolves resume skills with GLiNER + O*NET. Job postings are matched
against the same O*NET aliases with conservative phrase rules so that a posting's
skills receive the same concept IDs as resume skills:

* longest alias wins (``Amazon Web Services`` before ``Amazon``);
* only aliases that map to exactly one concept are used (ambiguous ones are skipped);
* a one-word alias that is also an ordinary English word (it appears in O*NET
  occupation/skill descriptions, e.g. "harvest", "analyze") must be written with
  a capital letter, and at the start of a sentence it must be a list item
  ("React, TypeScript" yes; "React to incidents" no);
* one-word generic O*NET skills ("Science", "Monitoring", "Writing") count only
  in "<skill> skills" form (e.g. "strong writing skills");
* single-letter aliases (C, R) must be upper-case;
* "Go" counts only inside a list ("Python, Go"), and aliases that end in punctuation
  ("ACT!") are skipped because the punctuation cannot be told apart from the word.

Mentions are classified as required or preferred from cue words in the same
sentence and from section headings. About-us and benefits sections are ignored.
"""

import json
import re
from dataclasses import dataclass, field

from .taxonomy import Concept, Taxonomy
from .text import IGNORED_KINDS, job_sections, normalize


_TOKEN = re.compile(r"[^\s/,;|()\[\]{}]+")
_STRIP_TAIL = ".:!?\"'”’*"
_STRIP_HEAD = "\"'“‘*•·-–—"
PREFERRED_CUE = re.compile(r"\b(preferred|nice[\s-]to[\s-]have|bonus|desired|desirable|ideally|"
                           r"(is |are |a |as a )plus|optional|familiarity with)\b", re.I)
SENTENCE_END = ".!?:"
MAX_ALIAS_TOKENS = 8
LIST_ONLY = frozenset({"go"})     # skill names that are everyday words even when capitalized


@dataclass(frozen=True)
class Token:
    key: str
    surface: str
    start: int
    end: int
    sentence_initial: bool
    next_char: str = ""    # first non-space character after the token ("\n" at line end, "" at text end)
    prev_char: str = ""    # last non-space character before the token


def surface_tokens(text: str) -> list[Token]:
    tokens = []
    for match in _TOKEN.finditer(text):
        raw, start = match.group(), match.start()
        head = len(raw) - len(raw.lstrip(_STRIP_HEAD))
        surface = raw[head:].rstrip(_STRIP_TAIL)
        if not surface:
            continue
        begin = start + head
        before = text[:begin].rstrip(" \t" + _STRIP_HEAD)
        initial = not before or before[-1] in "\n\r" or before[-1] in SENTENCE_END
        rest = text[begin + len(surface):].lstrip(" \t" + _STRIP_TAIL)
        prior = text[:begin].rstrip(" \t")
        tokens.append(Token(normalize(surface), surface, begin, begin + len(surface), initial, rest[:1], prior[-1:]))
    return tokens


_BOUNDARY = re.compile(r"[.!?;](?=\s)|\n")


def _sentence(text: str, start: int, end: int) -> str:
    """The sentence or list line around a span ("Node.js" does not end a sentence)."""
    before = [m.end() for m in _BOUNDARY.finditer(text, 0, start)]
    after = _BOUNDARY.search(text, end)
    return text[before[-1] if before else 0:after.start() if after else len(text)]


@dataclass(frozen=True)
class SkillMention:
    concept: Concept
    surface: str
    start: int
    end: int
    preferred: bool = False


@dataclass
class ExtractedSkills:
    required: list[Concept] = field(default_factory=list)
    preferred: list[Concept] = field(default_factory=list)
    mentions: list[SkillMention] = field(default_factory=list)


class SkillExtractor:
    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy
        self.phrases: dict[tuple[str, ...], Concept] = {}
        for alias, concepts in taxonomy.aliases["skill"].items():
            if len(concepts) != 1:
                continue  # never guess between several concepts
            if any(w.rstrip(_STRIP_TAIL) != w for w in normalize(alias).split()):
                continue  # e.g. "ACT!" would otherwise match the word "Act"
            key = tuple(t.key for t in surface_tokens(alias))
            if key and len(key) <= MAX_ALIAS_TOKENS:
                self.phrases[key] = concepts[0]
        self.max_tokens = max((len(k) for k in self.phrases), default=1)
        self.common_words = self._common_words(taxonomy)

    @staticmethod
    def _common_words(taxonomy: Taxonomy) -> frozenset[str]:
        words = set()
        for filename in ("occupations.jsonl", "skills.jsonl"):
            path = taxonomy.directory / filename
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    words.update(re.findall(r"[a-z][a-z'-]+", normalize(json.loads(line).get("description", ""))))
        return frozenset(words)

    def _accept_single(self, token: Token, concept: Concept, following: Token | None) -> bool:
        key, surface = token.key, token.surface
        if len(key) == 1:
            return surface.isupper() and not token.sentence_initial
        if concept.namespace == "onet:element":
            # Generic O*NET skills ("Science", "Monitoring") only in "<skill> skills" form.
            return following is not None and following.key in {"skill", "skills"}
        if key in LIST_ONLY:
            # "Go" is also an English verb ("Go the extra mile", "Go To Market"): only in a list.
            return token.prev_char in {",", "/", "|", ";"} or token.next_char in {",", "/", "|", ";", "("}
        if key not in self.common_words:
            return True  # distinctive product/language names such as docker, kubernetes
        if surface == surface.lower():
            return False
        # A capitalized common word starting a sentence counts only as a list item ("React, TypeScript").
        return not token.sentence_initial or token.next_char in {",", ";", "/", "|", "\n", "\r", ""}

    def find(self, text: str) -> list[SkillMention]:
        """All accepted mentions, longest match first, left to right."""
        tokens = surface_tokens(text or "")
        mentions, i = [], 0
        while i < len(tokens):
            matched = False
            for n in range(min(self.max_tokens, len(tokens) - i), 0, -1):
                concept = self.phrases.get(tuple(t.key for t in tokens[i:i + n]))
                if concept is None:
                    continue
                if n == 1 and not self._accept_single(tokens[i], concept,
                                                      tokens[i + 1] if i + 1 < len(tokens) else None):
                    continue
                first, last = tokens[i], tokens[i + n - 1]
                mentions.append(SkillMention(concept, text[first.start:last.end], first.start, last.end))
                i += n
                matched = True
                break
            if not matched:
                i += 1
        return mentions

    def extract_job_skills(self, text: str) -> ExtractedSkills:
        """Required/preferred skill concepts for one posting (each concept once)."""
        text = text or ""
        result = ExtractedSkills()
        sections = job_sections(text)
        seen_required, seen_preferred = {}, {}
        for mention in self.find(text):
            section = next((s for s in sections if s.start <= mention.start < s.end), None)
            if section is not None and section.kind in IGNORED_KINDS:
                continue
            preferred = (bool(PREFERRED_CUE.search(_sentence(text, mention.start, mention.end)))
                         or (section is not None and section.kind == "preferred"))
            mention = SkillMention(mention.concept, mention.surface, mention.start, mention.end, preferred)
            result.mentions.append(mention)
            (seen_preferred if preferred else seen_required).setdefault(mention.concept.id, mention.concept)
        result.required = list(seen_required.values())
        # Required wins when a skill is listed both ways.
        result.preferred = [c for cid, c in seen_preferred.items() if cid not in seen_required]
        return result
