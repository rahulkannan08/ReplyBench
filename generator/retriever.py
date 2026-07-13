"""
Few-shot example retriever for grounded reply generation.

Selects the most relevant past emails from the retrieval pool to use
as few-shot examples in the generation prompt. Uses a simple heuristic
similarity (no vector DB, no embeddings) appropriate for a 20-item dataset.
"""

from data.schemas import EmailScenario


def _similarity_score(target: EmailScenario, candidate: EmailScenario) -> float:
    """
    Compute heuristic similarity between two email scenarios.

    Priority order:
    1. Category match (highest — same category emails are most relevant)
    2. Sentiment match (secondary — similar tone context)
    3. Urgency match (tertiary)
    4. Thread context match (bonus)
    """
    score = 0.0

    # Category match — most important for relevant examples
    if target.category == candidate.category:
        score += 4.0

    # Sentiment match — tone context matters for reply style
    if target.sentiment == candidate.sentiment:
        score += 2.0

    # Urgency match
    if target.urgency == candidate.urgency:
        score += 1.0

    # Thread context match — if target has thread, prefer examples with thread
    if target.prior_thread_summary and candidate.prior_thread_summary:
        score += 0.5
    elif not target.prior_thread_summary and not candidate.prior_thread_summary:
        score += 0.25

    return score


def retrieve_few_shot_examples(
    target_email: EmailScenario,
    pool: list[EmailScenario],
    k: int = 3,
) -> list[EmailScenario]:
    """
    Select k most similar emails from the retrieval pool as few-shot examples.

    Args:
        target_email: The email to generate a reply for.
        pool: The retrieval pool (15 entries — NOT the held-out test set).
        k: Number of examples to retrieve.

    Returns:
        Up to k EmailScenario entries, sorted by descending similarity.
    """
    # Filter out the target itself (safety check)
    candidates = [e for e in pool if e.id != target_email.id]

    # Score and sort
    scored = [(c, _similarity_score(target_email, c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [c for c, _ in scored[:k]]
