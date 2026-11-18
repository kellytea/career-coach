"""Chat model access through ASU Voyager's OpenAI-compatible API (same settings as the parser).

Environment / .env: VOYAGER_API_KEY, VOYAGER_MODEL, optional VOYAGER_API_URL.
Replies are requested as JSON objects, which works with any OpenAI-compatible
endpoint and does not depend on native function calling.
"""

import json
import os
import re
from typing import Protocol


VOYAGER_API_URL = "https://openai.rc.asu.edu/v1"


class LLMError(RuntimeError):
    pass


class ChatModel(Protocol):
    def complete(self, messages: list[dict], *, temperature: float = 0.0) -> str:
        """Return the assistant message content for ``messages``."""


class VoyagerChatModel:
    def __init__(self, model: str | None = None, *, api_key: str | None = None, base_url: str | None = None,
                 max_tokens: int = 1200, extra: dict | None = None):
        from dotenv import load_dotenv

        load_dotenv()
        self.model = (model or os.getenv("VOYAGER_MODEL", "")).strip()
        if not self.model:
            raise LLMError("Set VOYAGER_MODEL in the environment or .env")
        self.api_key = api_key or os.getenv("VOYAGER_API_KEY")
        if not self.api_key:
            raise LLMError("Set VOYAGER_API_KEY in the environment or .env")
        self.base_url = base_url or os.getenv("VOYAGER_API_URL") or VOYAGER_API_URL
        self.max_tokens = max_tokens
        self.extra = dict(extra or {})   # e.g. {"reasoning_effort": "none"} for models that accept it
        self._client = None

    def complete(self, messages: list[dict], *, temperature: float = 0.0) -> str:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            completion = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=self.max_tokens,
                response_format={"type": "json_object"}, **self.extra)
        except Exception as error:  # network/auth/model errors surface as one type
            raise LLMError(f"Voyager request failed: {error}") from error
        content = completion.choices[0].message.content
        if not content:
            raise LLMError("Voyager returned an empty reply")
        return content


def parse_json(text: str) -> dict:
    """First JSON object in a reply, tolerating ```json fences and surrounding text."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    start = cleaned.find("{")
    if start == -1:
        raise LLMError("Reply contains no JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as error:
        raise LLMError(f"Reply is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise LLMError("Reply JSON is not an object")
    return value
