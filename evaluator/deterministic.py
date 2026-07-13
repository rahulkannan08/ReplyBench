"""
Deterministic scoring functions — no LLM calls needed.

Three dimensions scored purely from the text:
- conciseness: word count against ideal range
- tone_appropriateness: keyword/lexicon sentiment matching
- completeness: action-phrase detection
"""

from data.schemas import DimensionScore


# --- Conciseness ---

def score_conciseness(reply_body: str) -> DimensionScore:
    """
    Score reply conciseness based on word count.

    Ideal range: 50-200 words (appropriate for customer support).
    Score 10 if within range, linear degradation outside.
    """
    word_count = len(reply_body.split())

    if 50 <= word_count <= 200:
        value = 10.0
        detail = f"Word count {word_count} is within ideal range (50-200)"
    elif word_count < 50:
        # Too short — scale from 0 at 0 words to 10 at 50 words
        value = max(0.0, (word_count / 50) * 10)
        detail = f"Word count {word_count} is below ideal minimum (50)"
    else:
        # Too long — scale from 10 at 200 to 0 at 400+
        overage = word_count - 200
        value = max(0.0, 10.0 - (overage / 200) * 10)
        detail = f"Word count {word_count} exceeds ideal maximum (200)"

    return DimensionScore(
        value=round(value, 1),
        method="deterministic",
        detail=detail,
    )


# --- Tone Appropriateness ---

# Keyword sets for sentiment matching
APOLOGETIC_WORDS = {
    "sorry", "apologize", "apologies", "regret", "understand",
    "frustrating", "inconvenience", "appreciate your patience",
    "sincerely", "deeply",
}

EMPATHETIC_WORDS = {
    "understand", "hear you", "appreciate", "concern", "help",
    "assist", "resolve", "support", "together", "absolutely",
}

SOLUTION_WORDS = {
    "here's", "steps", "try", "check", "click", "navigate",
    "follow", "guide", "instructions", "let me explain",
    "simply", "easy",
}

WARM_WORDS = {
    "thank", "thanks", "glad", "great", "wonderful", "happy",
    "pleased", "welcome", "exciting", "fantastic", "awesome",
    "delighted",
}

PROFESSIONAL_WORDS = {
    "please", "kindly", "regarding", "further", "assist",
    "team", "review", "process", "confirm", "update",
}


def _count_keyword_hits(text: str, keywords: set[str]) -> int:
    """Count how many keywords appear in the text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def score_tone_appropriateness(
    reply_body: str, customer_sentiment: str
) -> DimensionScore:
    """
    Score tone match between reply and customer sentiment.

    Expected tone per sentiment:
    - angry -> apologetic + empathetic
    - confused -> solution-focused + clear
    - neutral -> professional + efficient
    - happy -> warm + professional
    """
    text_lower = reply_body.lower()

    if customer_sentiment == "angry":
        primary_hits = _count_keyword_hits(text_lower, APOLOGETIC_WORDS)
        secondary_hits = _count_keyword_hits(text_lower, EMPATHETIC_WORDS)
        expected = "apologetic + empathetic"
        # Need strong apologetic signals
        score = min(10.0, (primary_hits * 2.5) + (secondary_hits * 1.5))

    elif customer_sentiment == "confused":
        primary_hits = _count_keyword_hits(text_lower, SOLUTION_WORDS)
        secondary_hits = _count_keyword_hits(text_lower, EMPATHETIC_WORDS)
        expected = "clear + solution-focused"
        score = min(10.0, (primary_hits * 2.0) + (secondary_hits * 1.5))

    elif customer_sentiment == "happy":
        primary_hits = _count_keyword_hits(text_lower, WARM_WORDS)
        secondary_hits = _count_keyword_hits(text_lower, PROFESSIONAL_WORDS)
        expected = "warm + professional"
        score = min(10.0, (primary_hits * 2.5) + (secondary_hits * 1.0))

    else:  # neutral
        primary_hits = _count_keyword_hits(text_lower, PROFESSIONAL_WORDS)
        secondary_hits = _count_keyword_hits(text_lower, SOLUTION_WORDS)
        expected = "professional + efficient"
        score = min(10.0, (primary_hits * 2.0) + (secondary_hits * 1.5))

    detail = (
        f"Sentiment '{customer_sentiment}' expects {expected} tone. "
        f"Primary keyword hits: {primary_hits}, secondary: {secondary_hits}"
    )

    return DimensionScore(
        value=round(score, 1),
        method="deterministic",
        detail=detail,
    )


# --- Completeness ---

ACTION_PHRASES = [
    "next step", "i'll", "i will", "we'll", "we will",
    "follow up", "follow-up", "let me", "i'm going to",
    "will be", "expect to", "plan to", "schedule",
    "reach out", "get back to you", "update you",
    "resolve", "fix", "address", "look into",
    "send you", "provide you", "share with you",
    "please reply", "please let", "feel free",
    "click", "navigate", "go to", "visit",
    "contact", "call us", "email us",
]


def score_completeness(reply_body: str) -> DimensionScore:
    """
    Score whether the reply includes concrete action/next-step language.

    Checks for action-oriented phrases that indicate the reply provides
    a resolution path, not just acknowledgment.
    """
    text_lower = reply_body.lower()

    hits = [phrase for phrase in ACTION_PHRASES if phrase in text_lower]
    hit_count = len(hits)

    # Scale: 0 hits = 2 (basic acknowledgment), 1 = 5, 2 = 7, 3+ = 9-10
    if hit_count == 0:
        value = 2.0
        detail = "No action phrases detected -- reply may lack concrete next steps"
    elif hit_count == 1:
        value = 5.0
        detail = f"1 action phrase found: '{hits[0]}'"
    elif hit_count == 2:
        value = 7.0
        detail = f"2 action phrases found: {hits[:2]}"
    elif hit_count == 3:
        value = 9.0
        detail = f"3 action phrases found: {hits[:3]}"
    else:
        value = 10.0
        detail = f"{hit_count} action phrases found (showing top 4): {hits[:4]}"

    return DimensionScore(
        value=round(value, 1),
        method="deterministic",
        detail=detail,
    )
