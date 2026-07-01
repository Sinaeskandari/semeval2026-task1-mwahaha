"""Output cleaning and judge-verdict parsing.

Instruction-tuned models wrap their answers in chat-template scaffolding and
occasionally emit markdown or extra prose. These helpers extract the joke or
the JSON verdict robustly, mirroring the fallbacks used in the original scripts.
"""

from __future__ import annotations

import json
import re
from typing import Any

_CHINESE_RE = re.compile(r"[一-鿿]")


def contains_chinese(text: str) -> bool:
    """Return ``True`` if ``text`` contains any CJK Unified Ideograph."""
    return bool(_CHINESE_RE.search(text))


def extract_joke(raw_text: str, answer_split_token: str) -> str:
    """Isolate and clean a single joke from a decoded model response.

    Strategy (first match wins):
        1. Split on the ``"Joke:"`` anchor that every prompt ends with.
        2. Otherwise split on the model's chat-template token (``"model"`` for
           Gemma, ``"assistant"`` for Llama/Qwen).

    The tail is then trimmed to the first paragraph and stripped of quotes and
    stray template tokens.

    Args:
        raw_text: Full decoded output (prompt + completion).
        answer_split_token: Fallback marker preceding the answer.

    Returns:
        The cleaned joke text.
    """
    if "Joke:" in raw_text:
        joke = raw_text.split("Joke:")[-1]
    elif answer_split_token and answer_split_token in raw_text:
        joke = raw_text.split(answer_split_token)[-1]
    else:
        joke = raw_text

    # Remove residual chat-template tokens and normalise whitespace.
    joke = joke.replace("assistant", "").replace("model", "")
    joke = joke.strip()
    joke = joke.split("\n\n")[0]  # keep only the first paragraph
    return joke.strip().strip('"').strip("'").strip()


def parse_judge_verdict(response_text: str) -> dict[str, str]:
    """Parse a judge response into ``{"winner", "reasoning"}``.

    Handles the common failure modes: markdown code fences, trailing prose and
    malformed JSON. It first tries to decode the last ``{...}`` block, then
    falls back to regular expressions.

    Args:
        response_text: Raw judge output (already stripped of the prompt echo).

    Returns:
        Dict with ``winner`` in ``{"A", "B", "Tie", "Error"}`` and a
        ``reasoning`` string.
    """
    cleaned = re.sub(r"```json\s*", "", response_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned).strip()

    # 1. Try the last complete JSON object.
    end = cleaned.rfind("}")
    start = cleaned.rfind("{", 0, end) if end != -1 else -1
    if start != -1 and end != -1:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return {
                "winner": str(parsed.get("winner", "Error")).capitalize(),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
            }
        except json.JSONDecodeError:
            pass

    # 2. Regex fallback: take the last winner/reasoning match.
    winner = "Error"
    winner_matches = re.findall(
        r'winner["\']?\s*:\s*["\']?(A|B|Tie)["\']?', cleaned, re.IGNORECASE
    )
    if winner_matches:
        winner = winner_matches[-1].capitalize()

    reason_matches = re.findall(
        r'reasoning["\']?\s*:\s*["\']?(.*?)["\']?,?\s*\n',
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    reasoning = reason_matches[-1].strip() if reason_matches else cleaned[-200:].replace("\n", " ")
    return {"winner": winner, "reasoning": reasoning}


def normalise_row(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Extract ``(input_type, headline, word1, word2)`` from a raw TSV row.

    Reproduces the sentinel handling from the original loops: ``"-"``, empty
    strings and ``"nan"`` all mean "absent". A row is treated as a headline task
    when a usable headline is present, otherwise as a word-pair task.

    Returns:
        ``input_type`` is ``"headline"`` or ``"word-pair"``. For the word-pair
        case, missing words fall back to ``"something"`` / ``"random"``.
    """

    def clean(value: Any) -> str:
        text = str(value).strip()
        return "" if text in {"-", "", "nan", "NaN", "None"} or text.lower() == "nan" else text

    headline = clean(row.get("headline"))
    word1 = clean(row.get("word1"))
    word2 = clean(row.get("word2"))

    if headline:
        return "headline", headline, "", ""
    return "word-pair", "", word1 or "something", word2 or "random"
