"""
Hybrid evaluation engine -- orchestrates 3-tier scoring for each reply.

Tier 1: Reference comparison (semantic_similarity) -- LLM-based
Tier 2: Deterministic checks (conciseness, tone_appropriateness, completeness)
Tier 3: LLM-as-judge (relevance, correctness_faithfulness, safety_compliance)

Produces per-response scores + aggregate report.
"""

import os
import sys
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.schemas import (
    EmailScenario,
    GeneratedReply,
    DimensionScore,
    ResponseEvaluation,
    DimensionStats,
    AggregateReport,
)
from evaluator.deterministic import (
    score_conciseness,
    score_tone_appropriateness,
    score_completeness,
)
from evaluator.reference_compare import score_semantic_similarity
from evaluator.rubric import score_with_llm_judge


# Composite score weights -- justified by business impact
WEIGHTS = {
    "correctness_faithfulness": 0.25,  # Factual errors = real business/legal risk
    "relevance": 0.20,                # Must answer what was asked
    "safety_compliance": 0.15,         # Overpromising = liability
    "semantic_similarity": 0.15,       # Validates alignment with gold standard
    "completeness": 0.10,             # Customers need actionable next steps
    "tone_appropriateness": 0.10,     # Matters, but less catastrophic
    "conciseness": 0.05,             # Soft preference only
}

# Flag thresholds
FLAG_THRESHOLDS = {
    "low_relevance": ("relevance", 5.0),
    "hallucination_risk": ("correctness_faithfulness", 5.0),
    "safety_concern": ("safety_compliance", 5.0),
    "low_similarity": ("semantic_similarity", 4.0),
    "too_verbose": ("conciseness", 4.0),
    "tone_mismatch": ("tone_appropriateness", 4.0),
    "incomplete": ("completeness", 4.0),
}


def _evaluate_single(
    scenario: EmailScenario,
    reply: GeneratedReply,
) -> ResponseEvaluation:
    """Evaluate a single reply against its scenario across all 6 dimensions."""

    scores: dict[str, DimensionScore] = {}

    # --- Tier 2: Deterministic (fast, no API calls) ---
    scores["conciseness"] = score_conciseness(reply.reply_body)
    scores["tone_appropriateness"] = score_tone_appropriateness(
        reply.reply_body, scenario.sentiment
    )
    scores["completeness"] = score_completeness(reply.reply_body)

    # --- Tier 1: Reference comparison (1 LLM call) ---
    scores["semantic_similarity"] = score_semantic_similarity(
        reply.reply_body, scenario.reference_reply
    )

    # --- Tier 3: LLM-as-judge (1 batched LLM call for 3 dimensions) ---
    judge_scores = score_with_llm_judge(
        customer_email=scenario.customer_email_body,
        customer_subject=scenario.subject,
        category=scenario.category,
        sentiment=scenario.sentiment,
        urgency=scenario.urgency,
        generated_reply=reply.reply_body,
        prior_thread=scenario.prior_thread_summary,
        ideal_traits=scenario.ideal_reply_traits,
    )
    scores.update(judge_scores)

    # --- Composite score ---
    composite = sum(
        WEIGHTS[dim] * scores[dim].value
        for dim in WEIGHTS
        if dim in scores
    )

    # --- Flags ---
    flags = []
    for flag_name, (dim, threshold) in FLAG_THRESHOLDS.items():
        if dim in scores and scores[dim].value < threshold:
            flags.append(flag_name)

    return ResponseEvaluation(
        id=reply.id,
        scores=scores,
        composite_score=round(composite, 2),
        flags=flags,
    )


def evaluate_replies(
    scenarios: list[EmailScenario],
    replies: list[GeneratedReply],
    max_workers: int = 3,
) -> list[ResponseEvaluation]:
    """
    Evaluate all replies against their scenarios.

    Uses ThreadPoolExecutor for concurrent LLM evaluation calls.
    Each evaluation requires 2 LLM calls (similarity + judge), so we
    use fewer workers than generation to respect rate limits.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: EVALUATION")
    print(f"  Evaluating {len(replies)} replies across 6 dimensions")
    print(f"  Tiers: deterministic (3) + reference (1) + LLM-judge (3)")
    print(f"  Workers: {max_workers}")
    print("=" * 60)

    # Build scenario lookup
    scenario_map = {s.id: s for s in scenarios}

    evaluations: list[ResponseEvaluation] = []
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {}
        for reply in replies:
            scenario = scenario_map.get(reply.id)
            if not scenario:
                print(f"  [WARN] No scenario found for reply {reply.id}, skipping")
                continue
            future = executor.submit(_evaluate_single, scenario, reply)
            future_to_id[future] = reply.id

        for i, future in enumerate(as_completed(future_to_id), 1):
            reply_id = future_to_id[future]
            try:
                evaluation = future.result()
                evaluations.append(evaluation)
                print(
                    f"\n  [{i}/{len(future_to_id)}] {reply_id} -- "
                    f"composite={evaluation.composite_score:.2f} "
                    f"flags={evaluation.flags or 'none'}"
                )
            except Exception as e:
                failed.append(reply_id)
                print(f"\n  [{i}/{len(future_to_id)}] {reply_id} -- FAILED: {e}")

    # Sort by ID
    evaluations.sort(key=lambda e: e.id)

    print(f"\n  Evaluated: {len(evaluations)}/{len(replies)}")
    if failed:
        print(f"  Failed IDs: {failed}")

    return evaluations


def build_aggregate_report(
    evaluations: list[ResponseEvaluation],
    scenarios: list[EmailScenario],
) -> AggregateReport:
    """Build aggregate statistics from all evaluations."""
    if not evaluations:
        return AggregateReport(
            dimension_stats={},
            overall_composite_mean=0.0,
            overall_composite_median=0.0,
            worst_performing_category="N/A",
            best_performing_category="N/A",
            systemic_issues=["No evaluations completed"],
            total_evaluated=0,
            total_failed=0,
        )

    # --- Per-dimension stats ---
    dimension_values: dict[str, list[float]] = defaultdict(list)
    for ev in evaluations:
        for dim, score in ev.scores.items():
            dimension_values[dim].append(score.value)

    dimension_stats = {}
    for dim, values in dimension_values.items():
        dimension_stats[dim] = DimensionStats(
            mean=round(statistics.mean(values), 2),
            median=round(statistics.median(values), 2),
            min=round(min(values), 2),
            max=round(max(values), 2),
            std_dev=round(statistics.stdev(values) if len(values) > 1 else 0.0, 2),
        )

    # --- Composite stats ---
    composite_scores = [e.composite_score for e in evaluations]

    # --- Per-category analysis ---
    scenario_map = {s.id: s for s in scenarios}
    category_scores: dict[str, list[float]] = defaultdict(list)
    for ev in evaluations:
        scenario = scenario_map.get(ev.id)
        if scenario:
            category_scores[scenario.category].append(ev.composite_score)

    category_means = {
        cat: statistics.mean(scores) for cat, scores in category_scores.items()
    }

    worst_cat = min(category_means, key=category_means.get) if category_means else "N/A"
    best_cat = max(category_means, key=category_means.get) if category_means else "N/A"

    # --- Systemic issues detection ---
    systemic_issues = []

    for dim, stats in dimension_stats.items():
        if stats.mean < 5.0:
            systemic_issues.append(
                f"Low average {dim}: {stats.mean:.1f}/10 -- "
                f"indicates systematic weakness"
            )
        if stats.std_dev > 3.0:
            systemic_issues.append(
                f"High variance in {dim}: std_dev={stats.std_dev:.1f} -- "
                f"inconsistent performance"
            )

    # Check for sentiment-specific patterns
    sentiment_scores: dict[str, list[float]] = defaultdict(list)
    for ev in evaluations:
        scenario = scenario_map.get(ev.id)
        if scenario:
            tone_score = ev.scores.get("tone_appropriateness")
            if tone_score:
                sentiment_scores[scenario.sentiment].append(tone_score.value)

    for sentiment, scores in sentiment_scores.items():
        if scores and statistics.mean(scores) < 5.0:
            systemic_issues.append(
                f"Tone mismatch concentrated in '{sentiment}' sentiment entries "
                f"(avg tone score: {statistics.mean(scores):.1f})"
            )

    # Check for flag patterns
    flag_counts: dict[str, int] = defaultdict(int)
    for ev in evaluations:
        for flag in ev.flags:
            flag_counts[flag] += 1

    total = len(evaluations)
    for flag, count in flag_counts.items():
        if count > total * 0.5:  # More than half flagged
            systemic_issues.append(
                f"Frequent flag '{flag}': {count}/{total} responses flagged"
            )

    if not systemic_issues:
        systemic_issues.append("No systemic issues detected")

    return AggregateReport(
        dimension_stats=dimension_stats,
        overall_composite_mean=round(statistics.mean(composite_scores), 2),
        overall_composite_median=round(statistics.median(composite_scores), 2),
        worst_performing_category=worst_cat,
        best_performing_category=best_cat,
        systemic_issues=systemic_issues,
        total_evaluated=len(evaluations),
        total_failed=0,
    )
