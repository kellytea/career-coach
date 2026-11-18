"""Text utilities: lexical tokens, overlapping chunks and job-description sections."""

import re
import unicodedata
from dataclasses import dataclass


STOPWORDS = frozenset("""
a an and are as at be been but by can could do does for from had has have he her his i if in
into is it its may me more most must my no not of on or our ours she should so such than that
the their them then there these they this those to too us was we were what when where which while
who will with would you your yours about across all also any each etc per via within without
""".split())

# Keeps tokens such as c++, c#, node.js and 3d; splits on other punctuation.
_LEXICAL = re.compile(r"[a-z0-9][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lower-cased lexical tokens for TF-IDF/BM25."""
    tokens = _LEXICAL.findall(normalize(text))
    return tokens if keep_stopwords else [t for t in tokens if t not in STOPWORDS]


def chunk_words(text: str, max_words: int = 200, overlap: float = 0.1) -> list[str]:
    """Split text into word windows of ``max_words`` with fractional overlap."""
    if max_words < 1 or not 0 <= overlap < 1:
        raise ValueError("max_words must be >= 1 and 0 <= overlap < 1")
    words = (text or "").split()
    if not words:
        return []
    step = max(1, int(round(max_words * (1 - overlap))))
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
    return chunks


# ---------------------------------------------------------------------------
# Job description sections
# ---------------------------------------------------------------------------

SECTION_KINDS = (
    ("preferred", re.compile(r"\b(preferred|nice[\s-]to[\s-]have|bonus|desired|desirable|plus(es)?)\b", re.I)),
    ("requirements", re.compile(r"\b(requirements?|qualifications?|what you('ll)? (need|bring)|must[\s-]haves?|"
                                r"skills|experience|who you are|you have)\b", re.I)),
    ("responsibilities", re.compile(r"\b(responsibilit(y|ies)|duties|what you('ll)? do|the role|role overview|"
                                    r"day[\s-]to[\s-]day)\b", re.I)),
    ("benefits", re.compile(r"\b(benefits?|perks|compensation|salary|pay range|we offer|what we offer)\b", re.I)),
    ("about", re.compile(r"\b(about (us|the company|the team)|who we are|our (mission|company|story)|"
                         r"equal (employment )?opportunity|eeo|diversity)\b", re.I)),
)
IGNORED_KINDS = frozenset({"benefits", "about"})


@dataclass(frozen=True)
class Section:
    kind: str       # preferred | requirements | responsibilities | benefits | about | other
    heading: str
    start: int      # character offsets into the original text
    end: int

    def text(self, source: str) -> str:
        return source[self.start:self.end]


def _heading_kind(line: str) -> str | None:
    """Classify a short heading-like line, or return None if it is body text."""
    stripped = line.strip().strip("*#•-–— ").strip()
    if not stripped or len(stripped) > 70 or len(stripped.split()) > 8:
        return None
    if not (stripped.endswith(":") or stripped.isupper() or line.strip().startswith(("#", "**"))
            or (stripped.istitle() and not stripped.endswith("."))):
        return None
    for kind, pattern in SECTION_KINDS:
        if pattern.search(stripped):
            return kind
    return None


def job_sections(text: str) -> list[Section]:
    """Split a posting into headed sections; text before any heading is "other"."""
    text = text or ""
    sections, kind, heading, start, offset = [], "other", "", 0, 0
    for line in text.splitlines(keepends=True):
        found = _heading_kind(line)
        if found is not None:
            if offset > start:
                sections.append(Section(kind, heading, start, offset))
            kind, heading, start = found, line.strip(), offset
        offset += len(line)
    if offset > start or not sections:
        sections.append(Section(kind, heading, start, len(text)))
    return sections


def relevant_job_text(text: str) -> str:
    """Posting text without about-us/benefits/EEO sections (they dilute matching)."""
    kept = [s.text(text) for s in job_sections(text) if s.kind not in IGNORED_KINDS]
    joined = "".join(kept).strip()
    return joined or (text or "").strip()
