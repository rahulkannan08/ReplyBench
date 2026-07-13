"""
Reference reply comparison — semantic similarity scoring via LLM.

Single LLM call per response to assess how semantically similar the
generated reply is to the gold-standard reference reply.
"""

import json

from data.schemas import DimensionScore
from llm.client import call_llm
from pydantic import BaseModel, Field


class SimilarityJudgment(BaseModel):
    """LLM output schema for semantic similarity assessment."""
    score: float = Field(ge=0.0, le=10.0, description="Similarity score 0-10")
    justification: str = Field(description="One-sentence explanation")


def score_semantic_similarity(
    generated_reply: str,
    reference_reply: str,
) -> DimensionScore:
    """
    Score how semantically similar the generated reply is to the reference.

    Uses a single LLM call to assess meaning overlap (not exact wording).
    """
    prompt = f"""You are evaluating the semantic similarity between a generated customer support reply and a reference (gold-standard) reply.

Reference Reply (gold standard):
{reference_reply}

Generated Reply (to evaluate):
{generated_reply}

Rate the semantic similarity on a scale of 0-10:
- 0: Completely different meaning, addresses different topics
- 3: Shares some vague themes but misses key substance
- 5: Addresses the same topic but differs in important details or approach
- 7: Conveys most of the same meaning with minor differences
- 9: Nearly identical meaning despite different wording
- 10: Semantically identical

Focus on MEANING, not exact wording. Two replies can score 9-10 even if phrased very differently, as long as they convey the same substance, actions, and tone.

Respond with a JSON object containing "score" (float 0-10) and "justification" (one sentence)."""

    try:
        result = call_llm(prompt, response_schema=SimilarityJudgment)
        judgment = SimilarityJudgment.model_validate(json.loads(result))

        return DimensionScore(
            value=round(judgment.score, 1),
            method="reference_compare",
            justification=judgment.justification,
        )
    except Exception as e:
        # On failure, return a neutral score rather than crashing
        return DimensionScore(
            value=5.0,
            method="reference_compare",
            justification=f"Scoring failed, default applied: {e}",
        )
