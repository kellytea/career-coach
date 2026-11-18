"""Pinned O*NET retrieval; ambiguous links remain explicit candidates."""

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz, process

from .models import Candidate, Resolution

STOP_WORDS = {"and", "the", "of", "in", "to", "for", "with", "a", "an", "or", "on", "is", "are", "as"}


def tokens(value):
    return set(re.findall(r"[a-z][a-z0-9+#.]*", value.casefold())) - STOP_WORDS


def match_key(value):
    # Deliberately preserve C++, C#, C, .NET and punctuation in product names.
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


class Taxonomy:
    def __init__(self, directory, embedding_path=None):
        self.root = Path(directory)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.version = self.manifest["version"]
        self.records = {}
        self.aliases = {"occupation": defaultdict(set), "skill": defaultdict(set)}
        for name, kind in (("occupations.jsonl", "occupation"), ("skills.jsonl", "skill"),
                           ("software_examples.jsonl", "skill")):
            payload = (self.root / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != self.manifest["outputs"][name]["sha256"]:
                raise ValueError(f"Taxonomy checksum mismatch: {name}")
            for line in payload.decode("utf-8").splitlines():
                record = json.loads(line)
                if record["source_version"] != self.version:
                    raise ValueError("Mixed taxonomy releases")
                key = (record["namespace"], record["id"])
                self.records[key] = record
                for alias in {record["label"], *record.get("aliases", [])}:
                    self.aliases[kind][match_key(alias)].add(key)
        self.embedding_path = embedding_path
        self.embedder = None
        self.vectors = {}
        self.keys = {kind: list(index) for kind, index in self.aliases.items()}

    def _semantic(self, text, kind):
        if not self.embedding_path:
            return []
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            path = Path(self.embedding_path)
            if not path.is_dir():
                raise ValueError("EMBEDDING_MODEL_PATH must be a local model directory")
            self.embedder = SentenceTransformer(str(path), local_files_only=True, trust_remote_code=False)
        if kind not in self.vectors:
            keys = sorted({key for values in self.aliases[kind].values() for key in values})
            strings = [self.records[key]["label"] + ". " + self.records[key].get("description", "")
                       for key in keys]
            vectors = self.embedder.encode(strings, normalize_embeddings=True, show_progress_bar=False)
            self.vectors[kind] = keys, vectors
        keys, vectors = self.vectors[kind]
        query = self.embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = vectors @ query
        return [(keys[int(i)], float(scores[i])) for i in scores.argsort()[-5:][::-1]]

    def resolve(self, mention, context):
        kind = "occupation" if mention.label == "job title" else "skill"
        index = self.aliases[kind]
        key = match_key(mention.text)
        exact = sorted(index.get(key, set()))
        candidates = {}
        context_tokens = tokens(context) - tokens(mention.text)

        def add(record_key, method, score):
            record = self.records[record_key]
            if record_key not in candidates:
                definition = tokens(record["label"] + ". " + record.get("description", ""))
                overlap = len(context_tokens & definition) / max(1, len(context_tokens))
                candidates[record_key] = Candidate(
                    id=record["id"], namespace=record["namespace"], label=record["label"],
                    version=self.version, method=method, score=score,
                    description=record.get("description", ""), scores={"context_token_overlap": overlap})
            candidates[record_key].scores[method] = max(score, candidates[record_key].scores.get(method, float("-inf")))

        for record_key in exact:
            add(record_key, "exact_alias", 1.0)
        lexical, semantic = [], []
        if not exact and len(key) >= 3:
            for alias, score, _ in process.extract(key, self.keys[kind], scorer=fuzz.ratio, limit=5,
                                                    score_cutoff=55):
                for record_key in sorted(index[alias]):
                    add(record_key, "lexical_similarity", score / 100)
            lexical = sorted(candidates, key=lambda k: (-candidates[k].score,
                             -candidates[k].scores["context_token_overlap"], k))[:5]
            for record_key, score in self._semantic(mention.text + ". " + context[:800], kind):
                add(record_key, "embedding_cosine", score)
                semantic.append(record_key)
        # Preserve both retrieval channels instead of letting lexical ambiguity
        # push every semantic candidate off the end of a truncated list.
        if exact:
            ordered = sorted(exact, key=lambda k: (-candidates[k].scores["context_token_overlap"], k))
        else:
            ordered = list(dict.fromkeys([*lexical, *semantic]))
        ranked = [candidates[k] for k in ordered[:10]]
        unambiguous = len(exact) == 1 and len(key) >= 4 and mention.polarity == "positive"
        status = ("linked_exact" if unambiguous else "ambiguous" if len(exact) > 1
                  else "candidate" if ranked else "unresolved")
        return Resolution(mention_id=mention.id, status=status, candidates=ranked,
                          selected_id=ranked[0].id if unambiguous else None,
                          selected_namespace=ranked[0].namespace if unambiguous else None,
                          candidate_count=len(candidates),
                          reason="Unique full alias; lexical identity only" if unambiguous else
                          "Requires contextual review; no forced taxonomy assignment")
