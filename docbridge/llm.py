"""The three LLM jobs in the RAG loop: grade a chunk, rewrite a query, write a grounded answer.

GeminiLLM does them with Gemini. OfflineLLM does them with transparent lexical
rules so tests and keyless demos are deterministic. Both follow the same
protocol, so the LangGraph workflow does not know which one it is using.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol, Sequence

from .embed import normalize_tokens
from .ingest.chunker import split_sentences
from .models import Hit

IDK = "I could not find this in the documents you have access to."


class LLM(Protocol):
    def grade(self, question: str, text: str) -> bool: ...

    def rewrite(self, question: str, attempt: int) -> str: ...

    def generate(self, question: str, hits: Sequence[Hit]) -> str: ...


def format_context(hits: Sequence[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.ref}) {h.chunk.text}" for i, h in enumerate(hits, start=1))


# Offline

# Maps everyday wording to the vocabulary used in policy documents.
SYNONYMS = {
    "holiday": "annual leave days",
    "vacation": "annual leave days",
    "pto": "annual leave days",
    "time off": "annual leave days",
    "wfh": "remote work days",
    "work from home": "remote work days",
    "pay rise": "salary review merit increase",
    "raise": "salary review merit increase",
    "food": "meals per diem",
    "lunch money": "meals per diem",
    "allowance": "per diem",
    "reimbursed": "expense claims reimbursed payroll",
    "refund my expenses": "expense claims reimbursed",
    "laptop": "device security company laptops",
    "hacked": "security incident report",
    "phishing": "security incident report phishing",
    "outage": "incident severity SEV1",
    "site down": "incident severity SEV1",
    "postmortem": "postmortems blameless",
    "gift": "gifts hospitality declare compliance",
    "warehouse": "warehouse site",
    "flight": "booking travel flights class",
    "hotel": "hotels nightly cap",
    "length": "characters long",
    "minimum": "least",
    "tell": "report",
    "who do i": "report",
}


class OfflineLLM:
    def __init__(self, grade_threshold: float = 0.34):
        self.grade_threshold = grade_threshold

    @staticmethod
    def _expansions(question: str) -> list[str]:
        lowered = question.lower()
        return [v for k, v in SYNONYMS.items() if re.search(rf"\b{re.escape(k)}\b", lowered)]

    def grade(self, question: str, text: str) -> bool:
        """Relevant if enough of the question's terms, or their policy wording, appear in the chunk."""
        doc = set(normalize_tokens(text))
        terms = list(dict.fromkeys(normalize_tokens(question)))
        if not terms:
            return False
        matched = sum(1 for t in terms if t in doc)
        bonus = sum(1 for t in dict.fromkeys(normalize_tokens(" ".join(self._expansions(question)))) if t in doc and t not in terms)
        return (matched + 0.5 * bonus) / len(terms) >= self.grade_threshold

    def rewrite(self, question: str, attempt: int) -> str:
        expansions = self._expansions(question)
        keywords = " ".join(dict.fromkeys(normalize_tokens(question)))
        if attempt == 0 and expansions:
            return f"{keywords} {' '.join(expansions)}"
        return keywords if attempt == 0 else f"{keywords} {' '.join(expansions)} policy"

    def generate(self, question: str, hits: Sequence[Hit]) -> str:
        q = set(normalize_tokens(self.rewrite(question, 0)))
        scored = []
        for i, hit in enumerate(hits, start=1):
            for sentence in split_sentences(hit.chunk.text):
                toks = normalize_tokens(sentence)
                if len(toks) < 4:  # skip headings
                    continue
                scored.append((sum(1 for t in toks if t in q) / (len(toks) ** 0.5), i, sentence))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = [s for s in scored if s[0] > 0][:2]
        # Keep a second sentence only if it is nearly as relevant as the first.
        best = [s for s in best if s[0] >= 0.7 * best[0][0]] if best else best
        if not best:
            return IDK
        return " ".join(f"{sentence} [{i}]" for _, i, sentence in best)


# Gemini

GRADE_PROMPT = """You are grading retrieval results for a question answering system.
Question: {question}
Passage: {text}
Does the passage contain information that helps answer the question? Reply as JSON: {{"relevant": true}} or {{"relevant": false}}."""

REWRITE_PROMPT = """The search for this question found nothing relevant in a company's policy documents.
Rewrite it as a short search query using the formal wording a policy document would use.
Return only the query.
Question: {question}"""

ANSWER_PROMPT = """Answer the question using only the numbered context passages.
Cite every sentence with the passage number in square brackets, like [1].
If the context does not contain the answer, reply exactly: {idk}
Keep the answer under 80 words.

Context:
{context}

Question: {question}"""


class GeminiLLM:
    def __init__(self, model: str | None = None):
        from google import genai

        self.client = genai.Client()
        self.model = model or os.environ.get("DOCBRIDGE_LLM_MODEL", "gemini-2.5-flash")

    def _text(self, prompt: str, json_mode: bool = False) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json" if json_mode else "text/plain",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        return (self.client.models.generate_content(model=self.model, contents=prompt, config=config).text or "").strip()

    def grade(self, question: str, text: str) -> bool:
        try:
            return bool(json.loads(self._text(GRADE_PROMPT.format(question=question, text=text), json_mode=True))["relevant"])
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def rewrite(self, question: str, attempt: int) -> str:
        return self._text(REWRITE_PROMPT.format(question=question)) or question

    def generate(self, question: str, hits: Sequence[Hit]) -> str:
        return self._text(ANSWER_PROMPT.format(idk=IDK, context=format_context(hits), question=question)) or IDK


def default_llm() -> LLM:
    from .embed import has_gemini_credentials

    choice = os.environ.get("DOCBRIDGE_LLM")
    if choice == "offline" or (choice is None and not has_gemini_credentials()):
        return OfflineLLM()
    return GeminiLLM()
