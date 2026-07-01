"""Prompt templates for base generation, RAG generation and judging.

The prompts follow the few-shot, constraint-driven design described in the
paper: each competitor is instructed to act as a "witty, cynical stand-up
comedian" and must satisfy a task constraint (satirise a *headline* or include
a *word pair*). The judge prompt implements the paired-comparison protocol used
for the round-robin tournament.
"""

from __future__ import annotations

from typing import Literal

InputType = Literal["headline", "word-pair"]

# Few-shot exemplars reused across the base and RAG prompts. Keeping them in one
# place ensures every model is conditioned on identical demonstrations.
_HEADLINE_EXAMPLES = """Headline: "Study finds 90% of office meetings could be emails."
Joke: "A new study found that 90% of office meetings could be emails, which implies the other 10% could have just been silence."

Headline: "Billionaire builds giant clock inside a mountain."
Joke: "A billionaire is building a giant clock inside a mountain, finally providing a way to tell time for the five people who actually survive the apocalypse."

Headline: "Scientists discover new species of deep-sea jelly."
Joke: "Scientists have discovered a new species of jelly at the bottom of the ocean, mostly because they were tired of looking for the ones in their donuts.\""""

_WORD_EXAMPLES = """Words: "unplug" + "fridge"
Joke: "My current relationship is exactly like an unplugged fridge: it's cold, dark, and I'm terrified to open it and see what's rotting inside."

Words: "hammer" + "banana"
Joke: "I tried to fix my diet with the same tool I use to fix my furniture, but it turns out taking a hammer to a banana just makes a smoothie with too much crunch."

Words: "measure" + "pizza"
Joke: "Trying to measure happiness with money is like trying to measure a pizza with a thermometer: you're using the wrong tool and you're just going to burn your hand.\""""

_ENGLISH_RULE = "\n- Output MUST be in English. No Chinese characters allowed."


def build_base_prompt(
    input_type: InputType,
    *,
    headline: str = "",
    word1: str = "",
    word2: str = "",
    enforce_english: bool = False,
) -> str:
    """Build the base (no-retrieval) generation prompt.

    Args:
        input_type: ``"headline"`` or ``"word-pair"``.
        headline: Headline text (used when ``input_type == "headline"``).
        word1, word2: The two concepts to connect (word-pair case).
        enforce_english: Append an explicit English-only rule (used for Qwen).

    Returns:
        The fully formatted prompt string.
    """
    english = _ENGLISH_RULE if enforce_english else ""
    if input_type == "headline":
        return f"""### Instruction
You are a witty, cynical stand-up comedian.
Your task is to write EXACTLY ONE punchy joke (1-2 sentences) based on the provided headline.

### Examples
{_HEADLINE_EXAMPLES}

### Task
Headline: "{headline}"

### Constraints
1. The joke must be STANDALONE. Do not assume the audience has read the headline; include the premise in the joke itself.
2. Be clever, cynical, or ironic.
3. NO explanations or conversational filler (e.g., do not write "Here is the joke").
4. Output ONLY the joke.{english}

### Response
Joke:"""

    return f"""You are a witty, cynical stand-up comedian.

Task: Write EXACTLY ONE punchy joke (1-2 sentences) that connects the following two concepts: "{word1}" and "{word2}".

Here are examples of how to connect random words creatively:

{_WORD_EXAMPLES}

MANDATORY Rules:
- You can use the words literally OR metaphorically.
- The logic must hold up (e.g., do not say a laptop cooks food).
- Do NOT explain the joke.
- Do NOT use filler like "Here is a joke."{english}

Words: "{word1}", "{word2}"
Joke:"""


def build_rag_prompt(
    input_type: InputType,
    context: str,
    *,
    headline: str = "",
    word1: str = "",
    word2: str = "",
) -> str:
    """Build the retrieval-augmented generation prompt.

    The retrieved Wikipedia context is supplied as optional inspiration; the
    model is explicitly told not to quote or summarise it.

    Args:
        input_type: ``"headline"`` or ``"word-pair"``.
        context: Retrieved background text.
        headline: Headline text (headline case).
        word1, word2: The two concepts to include (word-pair case).

    Returns:
        The fully formatted RAG prompt string.
    """
    if input_type == "headline":
        return f"""### Instruction
You are a witty, cynical stand-up comedian. Write ORIGINAL humor (do not reuse or paraphrase known jokes).
Use the Background Facts only if they help inspire the joke. Do not quote them.

Rules:
- Output EXACTLY ONE joke (1-2 sentences).
- The joke must be STANDALONE: include the premise so it makes sense without reading the headline.
- Be clever, cynical, or ironic; end with a twist if possible.
- Do NOT explain the joke.
- Do NOT summarize the headline. Make it a joke.
- Keep it punchy (max ~35 words).

Background Facts (optional):
{context}

### Examples (style only)
{_HEADLINE_EXAMPLES}

### Task
Headline: "{headline}"

### Response
Joke:"""

    return f"""### Instruction
You are a witty, cynical stand-up comedian. Write ORIGINAL humor (do not reuse or paraphrase known jokes).
Use the Background Facts only if they help inspire the joke. Do not quote them.

Rules:
- Output EXACTLY ONE joke (1-2 sentences).
- Must include BOTH words (case-insensitive is OK): "{word1}" and "{word2}".
- Be clever, cynical, or ironic; end with a twist if possible.
- No explanations, no analysis, no extra text.
- Keep it punchy (max ~35 words).

Background Facts (optional):
{context}

### Examples (style only)
Words: "unplug" + "fridge"
Joke: "My current relationship is like an unplugged fridge: cold, dark, and I'm scared to open it and see what's rotting inside."

### Task
Words: "{word1}", "{word2}"

### Response
Joke:"""


def build_judge_prompt(
    input_type: InputType,
    context: str,
    joke_a: str,
    joke_b: str,
    *,
    name_a: str = "A",
    name_b: str = "B",
) -> str:
    """Build the paired-comparison prompt for the LLM judge.

    The judge first checks constraint satisfaction, then compares humour, and
    returns a strict JSON verdict. The English-only reminder mitigates the same
    code-switching issue seen during generation.

    Args:
        input_type: ``"headline"`` or ``"word-pair"``.
        context: Headline text, or the comma-separated word pair.
        joke_a, joke_b: The two candidate jokes.
        name_a, name_b: Competitor names shown to the judge (for logging only).

    Returns:
        The fully formatted judge prompt string.
    """
    if input_type == "headline":
        constraint = "The joke must satirize or directly relate to the Input Content headline."
    else:
        constraint = f"The joke MUST contain the exact words: {context}"

    return f"""You are an impartial humor judge. You evaluate AI-generated jokes based on quality and adherence to specific constraints.
I will provide you with a context and two jokes (Joke A and Joke B).
Your goal is to choose the winner based on two criteria:
1. CONSTRAINT CHECK: Does the joke follow the specific rule provided below?
2. QUALITY: Which joke is funnier, more clever, or better written?

### CONTEXT
Constraint Rule: {constraint}
Input Content: {context}

### THE CANDIDATES
Joke A ({name_a}):
{joke_a}

Joke B ({name_b}):
{joke_b}

### EVALUATION STEPS
1. Analyze if Joke A satisfies the Constraint Rule.
2. Analyze if Joke B satisfies the Constraint Rule.
3. Compare the humor and wit of both jokes.
4. If one joke follows the rule and the other does not, the one that follows the rule MUST win.
5. If both follow the rule (or both fail), pick the funniest one.

### OUTPUT FORMAT
Provide your final decision in the following JSON format. Do not output any other text or markdown.
Keeping the reasoning concise (MAXIMUM 1 SENTENCE).
**IMPORTANT: Write ONLY in English. Do not generate Chinese.**

{{
  "reasoning": "A single English sentence explaining the winner.",
  "winner": "A" or "B" or "Tie"
}}"""
