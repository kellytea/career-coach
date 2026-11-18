import pymupdf
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

class ResumeParser:
    # =========================
    # CONFIG
    # =========================
    FEATURE_HEADERS = {
        "title": ["title", "role", "position"],
        "summary": ["summary", "professional summary", "career summary", "profile", "professional profile"],
        "experience": [
            "experience", "work experience", "professional experience",
            "work history", "employment history"
        ],
        "skills": ["skills", "technical skills", "core skills", "highlights", "skill highlights"],
        "education": ["education", "education and training", "academic background", "training"],
        "projects": ["projects", "project experience", "portfolio", "technical projects"],
    }

    MONTHS_PATTERN = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"

    SKILL_LABELS = {
        "languages", "programming languages", "frameworks", "libraries", "tools",
        "frameworks/tools", "frameworks and tools", "databases", "platforms",
        "cloud platforms", "technologies",
    }


    # =========================
    # CLEANING
    # =========================
    def clean_line(self, text: str, *, preserve_commas: bool = False) -> str:
        if not isinstance(text, str):
            return ""

        text = re.sub(r"[\u2022\u25cf\u25cb\u25aa\uf0b7\xb7\*\•]", " , ", text)
        text = text.encode("ascii", errors="ignore").decode()
        text = text.lower().replace("&", " and ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"( , \s*)+", ", ", text)

        if preserve_commas:
            return text.strip().lstrip(",").strip()
        return text.strip().strip(",")


    # =========================
    # HEADER LOGIC
    # =========================
    def is_header_line(self, line: str) -> bool:
        if not line or len(line) > 40:
            return False
        if line.endswith(".") or len(line.split()) > 5:
            return False
        return True

    def match_header(self, line: str):
        for feature, headers in self.FEATURE_HEADERS.items():
            for h in headers:
                if line == h or line.startswith(h + ":"):
                    return feature
        return None


    # =========================
    # TITLE FALLBACK
    # =========================
    def infer_title_from_experience(self, experience: str) -> str:
        if not experience:
            return ""
        first = re.sub(r"\b\d{4}\b", "", experience.split("|")[0])
        return " ".join(first.split()[:4])


    # =========================
    # PDF
    # =========================
    def pdf_to_text(self, pdf_path: Path) -> str:
        doc = pymupdf.open(pdf_path)
        return "\n".join(p.get_text("text") for p in doc)


    # =========================
    # FEATURE EXTRACTION
    # =========================
    def is_contact_line(self, line: str) -> bool:
        return bool(re.search(r"https?://|www\.|\S+@\S+|(?:linkedin|github)\.com", line)
                    or re.fullmatch(r"[+\d\s().|/-]+", line) is not None
                    and len(re.sub(r"\D", "", line)) >= 7)

    def starts_unlabeled_summary(self, lines, index: int) -> bool:
        line = self.clean_line(lines[index])
        if self.is_contact_line(line):
            return False
        # Long prose can be recognized without punctuation. For shorter PDF
        # lines, inspect a small continuation window to handle wrapped sentences.
        if len(line.split()) >= 10:
            return True
        if len(line.split()) < 4:
            return False
        paragraph = [line]
        for raw in lines[index + 1:index + 3]:
            continuation = self.clean_line(raw)
            label = continuation.partition(":")[0].strip()
            if not continuation or self.is_contact_line(continuation) or self.match_header(label):
                break
            paragraph.append(continuation)
        prose = " ".join(paragraph)
        return len(prose.split()) >= 8 and bool(re.search(r"[.!?](?:\s|$)", prose))

    def extract_features(self, text: str) -> dict:
        lines = text.split("\n")
        data = {k: "" for k in self.FEATURE_HEADERS}
        section_texts = {k: [] for k in self.FEATURE_HEADERS}
        current = "title"
        title_taken = False
        first_section_seen = False

        for index, raw in enumerate(lines):
            line = self.clean_line(raw, preserve_commas=current == "education")
            if not line:
                continue

            # Judge the label, not the length of its inline content. Keep the
            # content after ':' so short lists are not discarded as headings.
            label, colon, content = line.partition(":")
            label = label.strip()
            matched = self.match_header(label) if self.is_header_line(label) else None
            if matched:
                if matched != "title":
                    first_section_seen = True
                current = matched
                line = content.strip() if colon else ""
            elif current == "skills":
                skill_label = re.sub(r"\s*/\s*", "/", label)
                if skill_label in self.SKILL_LABELS:
                    line = content.strip() if colon else ""
            if not line:
                continue

            if current == "title" and not title_taken:
                data["title"] = line
                section_texts[current].append(raw.strip() if not matched else raw.partition(":")[2].strip())
                title_taken = True
                continue

            if (not first_section_seen and current == "title" and not matched
                    and self.starts_unlabeled_summary(lines, index)):
                current = "summary"

            # Contact details may also follow the introductory paragraph.
            if current == "summary" and self.is_contact_line(line):
                continue

            source_line = raw.strip()
            if matched and colon:
                source_line = raw.partition(":")[2].strip()
            elif current == "skills" and colon and skill_label in self.SKILL_LABELS:
                source_line = raw.partition(":")[2].strip()
            section_texts[current].append(source_line)

            if current == "skills":
                sep = ", " if data[current] else ""
                data[current] += sep + line
            elif current == "education":
                sep = "\n" if data[current] else ""
                data[current] += sep + line
            elif current == "projects":
                sep = "; " if data[current] else ""
                data[current] += sep + line
            else:
                data[current] += " " + line

        return {**{k: v.strip() for k, v in data.items()},
                "section_texts": {k: "\n".join(v) for k, v in section_texts.items()}}


    # =========================
    # REFINE EDUCATION
    # =========================
    def refine_education(self, sections: dict) -> dict:
        refined = self._education_blocks(sections)
        for category in ("coursework", "awards"):
            refined[category] = [" ".join(block["text"].split()) for block in refined[category]]
        return refined

    def _education_blocks(self, sections: dict) -> dict:
        """Keep source spans internally while exposing clean item lists publicly."""
        text = sections.get("education", "")
        result = {"degree_entries": [], "coursework": [], "awards": [], "context": []}
        degree_pattern = re.compile(
            r"\b(?:[bm]\.(?:s|a)\.?|(?:bs|ms|ba|ma|bsc|msc|phd)\b|ph\.d\.?|"
            r"(?:bachelor|master|associate)(?:s|'s)?\b|doctorate\b|diploma\b)", re.I
        )
        labels = {
            "coursework": r"^(?:relevant\s+)?(?:coursework|course work|courses)\s*(?::|$)",
            "awards": r"^(?:awards?(?:\s+(?:and|&)\s+scholarships?)?|scholarships?(?:\s+(?:and|&)\s+awards?)?|honors?)\s*(?::|$)",
        }
        active = None
        offset = 0
        for raw in text.splitlines(keepends=True):
            line = raw.strip()
            start = offset + len(raw) - len(raw.lstrip())
            offset += len(raw)
            if not line:
                continue
            category = None
            for kind, pattern in labels.items():
                match = re.match(pattern, line, re.I)
                if match:
                    category = kind
                    start += match.end()
                    line = text[start:offset].strip()
                    start = text.find(line, start, offset) if line else offset
                    break
            if category:
                active = {"text": line, "start": start, "end": start + len(line)}
                result[category].append(active)
            elif degree_pattern.search(line) and not re.search(r"\baccelerated\b", line, re.I):
                active = {"text": line, "start": start, "end": start + len(line)}
                result["degree_entries"].append(active)
            elif active is not None:
                active["end"] = start + len(line)
                active["text"] = text[active["start"]:active["end"]]
            else:
                result["context"].append({"text": line, "start": start, "end": start + len(line)})
        for category in ("coursework", "awards"):
            items = []
            for block in result[category]:
                for match in re.finditer(r"[^,]+", block["text"]):
                    value = match[0].strip()
                    if not value:
                        continue
                    start = block["start"] + match.start() + len(match[0]) - len(match[0].lstrip())
                    items.append({"text": value, "start": start, "end": start + len(value)})
            result[category] = items
        return result


    # =========================
    # REFINE EXPERIENCE
    # =========================
    def refine_experience(self, sections: dict) -> list:
        """Heuristically split date-led job headers; keep undated text together."""
        text = sections.get("section_texts", {}).get("experience", sections.get("experience", ""))
        lines = []
        offset = 0
        for raw in text.splitlines(keepends=True):
            if raw.strip():
                lines.append((offset, raw.strip()))
            offset += len(raw)
        endpoint = rf"(?:(?:{self.MONTHS_PATTERN})\b(?:\s+\d{{4}})?|\d{{1,2}}/\d{{4}}|\d{{4}})"
        date_range = re.compile(rf"{endpoint}\s*(?:[-–—�]|\bto\b)\s*(?:{endpoint}|present\b|current\b)", re.I)
        starts = {0} if text.strip() else set()
        for i, (start, line) in enumerate(lines):
            match = date_range.search(line)
            if not match or len(line) > 180 or line.startswith(("•", "-", "*", "�")) or line.endswith("."):
                continue
            if not line[:match.start()].strip() and i:
                previous_start, previous = lines[i - 1]
                if len(previous) <= 100 and not previous.endswith(".") and not previous.startswith(("•", "-", "*", "�")):
                    start = previous_start
            starts.add(start)
        boundaries = sorted(starts) + [len(text)]
        return [{"text": text[start:end].rstrip(), "start": start,
                    "end": start + len(text[start:end].rstrip())}
                for start, end in zip(boundaries, boundaries[1:]) if text[start:end].strip()]


    # =========================
    # PIPELINE
    # =========================
    def run(self, pdf_folder: str) -> pd.DataFrame:
        rows = []

        for pdf in Path(pdf_folder).glob("*.pdf"):
            text = self.pdf_to_text(pdf)
            feat = self.extract_features(text)
            feat["cv_id"] = pdf.name

            for k in feat:
                if k not in ("education", "section_texts"):
                    feat[k] = self.clean_line(feat[k])

            if not feat["title"]:
                feat["title"] = self.infer_title_from_experience(feat["experience"])

            feat["skills_list"] = [s.strip() for s in feat["skills"].split(",") if s.strip()]
            feat["education_refined"] = self.refine_education({"education": feat["section_texts"]["education"]})
            feat["experience_refined"] = self.refine_experience(feat)
            rows.append(feat)

        return pd.DataFrame(rows)
