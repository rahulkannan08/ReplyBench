"""
Metric validation script -- proves the evaluation metric discriminates
between good and bad replies.

Case A: Score each reference reply against itself (should score ~9-10)
Case B: Score deliberately bad control replies (should score ~2-4)

The gap between Case A and Case B demonstrates the metric reflects
real quality, not just assigning numbers.
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.schemas import EmailScenario, GeneratedReply
from evaluator.evaluate import _evaluate_single, WEIGHTS

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "results", "metric_validation_report.json")


def _make_bad_reply(scenario: EmailScenario) -> GeneratedReply:
    """Create a deliberately bad reply for a scenario."""
    # Bad replies: wrong tone, off-topic, no next steps, overpromises
    bad_templates = {
        "angry": (
            "Thanks for your email! Everything is great on our end. "
            "Have you tried turning it off and on again? "
            "We're sure it's probably your fault. Cheers!"
        ),
        "confused": (
            "Per our policy section 47.3.b, the aforementioned functionality "
            "is deprecated as of Q3. Please consult the 200-page user manual "
            "for further clarification on this matter."
        ),
        "neutral": (
            "I totally understand your frustration and I'm SO sorry! "
            "This must be incredibly upsetting for you. I promise we will "
            "give you a full refund of $5,000 and a free lifetime subscription! "
            "Our CEO will personally call you tomorrow at 3pm!"
        ),
        "happy": (
            "We don't support that feature and have no plans to. "
            "You should probably switch to a competitor. "
            "Let us know if you need help cancelling your account."
        ),
    }

    bad_body = bad_templates.get(scenario.sentiment, bad_templates["neutral"])

    return GeneratedReply(
        id=scenario.id,
        reply_subject=f"Re: {scenario.subject}",
        reply_body=bad_body,
        tone_used="inappropriate",
        suggested_next_action="none",
        confidence=0.5,
    )


def _make_reference_as_reply(scenario: EmailScenario) -> GeneratedReply:
    """Use the reference reply as the 'generated' reply (should score high)."""
    return GeneratedReply(
        id=scenario.id,
        reply_subject=f"Re: {scenario.subject}",
        reply_body=scenario.reference_reply,
        tone_used="appropriate",
        suggested_next_action="as specified in reference",
        confidence=0.95,
    )


def main():
    start_time = time.time()

    print("\n" + "=" * 70)
    print("METRIC VALIDATION")
    print("  Case A: Reference reply scored against itself (expect ~9-10)")
    print("  Case B: Deliberately bad control replies (expect ~2-5)")
    print("=" * 70)

    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenarios = [EmailScenario.model_validate(entry) for entry in data]

    # Use 3 diverse scenarios for validation (save API calls)
    # Pick different sentiments for coverage
    validation_scenarios = []
    seen_sentiments = set()
    for s in scenarios:
        if s.sentiment not in seen_sentiments and len(validation_scenarios) < 3:
            validation_scenarios.append(s)
            seen_sentiments.add(s.sentiment)

    case_a_scores = []
    case_b_scores = []
    case_details = []

    for i, scenario in enumerate(validation_scenarios, 1):
        print(f"\n  --- Scenario {i}/3: {scenario.id} ({scenario.category}/{scenario.sentiment}) ---")

        # Case A: reference reply
        print(f"  Case A: Scoring reference reply against itself...")
        ref_reply = _make_reference_as_reply(scenario)
        eval_a = _evaluate_single(scenario, ref_reply)
        case_a_scores.append(eval_a.composite_score)
        print(f"  Case A composite: {eval_a.composite_score:.2f}")

        # Brief pause to respect rate limits between eval calls
        time.sleep(8)

        # Case B: bad reply
        print(f"  Case B: Scoring deliberately bad reply...")
        bad_reply = _make_bad_reply(scenario)
        eval_b = _evaluate_single(scenario, bad_reply)
        case_b_scores.append(eval_b.composite_score)
        print(f"  Case B composite: {eval_b.composite_score:.2f}")

        # Pause before next scenario
        if i < len(validation_scenarios):
            time.sleep(8)

        gap = eval_a.composite_score - eval_b.composite_score
        print(f"  Gap: {gap:+.2f}")

        case_details.append({
            "scenario_id": scenario.id,
            "category": scenario.category,
            "sentiment": scenario.sentiment,
            "case_a_composite": eval_a.composite_score,
            "case_a_scores": {k: v.value for k, v in eval_a.scores.items()},
            "case_b_composite": eval_b.composite_score,
            "case_b_scores": {k: v.value for k, v in eval_b.scores.items()},
            "gap": round(gap, 2),
        })

    # Aggregate
    mean_a = sum(case_a_scores) / len(case_a_scores)
    mean_b = sum(case_b_scores) / len(case_b_scores)
    mean_gap = mean_a - mean_b

    report = {
        "summary": {
            "case_a_mean_composite": round(mean_a, 2),
            "case_b_mean_composite": round(mean_b, 2),
            "mean_gap": round(mean_gap, 2),
            "validation_passed": mean_gap >= 2.0,
        },
        "cases": case_details,
        "weights_used": WEIGHTS,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"VALIDATION RESULTS")
    print(f"  Case A (reference) mean composite: {mean_a:.2f}")
    print(f"  Case B (bad reply) mean composite:  {mean_b:.2f}")
    print(f"  Mean gap:                           {mean_gap:+.2f}")
    print(f"  Validation passed (gap >= 2.0):     {'YES' if mean_gap >= 2.0 else 'NO'}")
    print(f"  Report: {REPORT_PATH}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
