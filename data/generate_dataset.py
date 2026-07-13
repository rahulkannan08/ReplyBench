"""
Generate 20 synthetic customer email + reference reply pairs via LLM.

Produces data/dataset.json with variety across:
- 6 categories (billing, bug_report, refund_request, feature_question, cancellation, complaint)
- 4 sentiments (angry, neutral, confused, happy)
- 3 urgency levels (low, medium, high)
- ~40% with prior_thread_summary
"""

import json
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.client import call_llm
from data.schemas import EmailScenario

# Pre-defined scenario specs to ensure coverage
SCENARIO_SPECS = [
    # billing (3 entries)
    {"id": "email_001", "category": "billing", "sentiment": "angry", "urgency": "high", "has_thread": True},
    {"id": "email_002", "category": "billing", "sentiment": "confused", "urgency": "medium", "has_thread": False},
    {"id": "email_003", "category": "billing", "sentiment": "neutral", "urgency": "low", "has_thread": False},
    # bug_report (3 entries)
    {"id": "email_004", "category": "bug_report", "sentiment": "angry", "urgency": "high", "has_thread": True},
    {"id": "email_005", "category": "bug_report", "sentiment": "confused", "urgency": "medium", "has_thread": False},
    {"id": "email_006", "category": "bug_report", "sentiment": "neutral", "urgency": "high", "has_thread": True},
    # refund_request (3 entries)
    {"id": "email_007", "category": "refund_request", "sentiment": "angry", "urgency": "high", "has_thread": False},
    {"id": "email_008", "category": "refund_request", "sentiment": "neutral", "urgency": "medium", "has_thread": True},
    {"id": "email_009", "category": "refund_request", "sentiment": "happy", "urgency": "low", "has_thread": False},
    # feature_question (4 entries)
    {"id": "email_010", "category": "feature_question", "sentiment": "happy", "urgency": "low", "has_thread": False},
    {"id": "email_011", "category": "feature_question", "sentiment": "confused", "urgency": "medium", "has_thread": True},
    {"id": "email_012", "category": "feature_question", "sentiment": "neutral", "urgency": "low", "has_thread": False},
    {"id": "email_013", "category": "feature_question", "sentiment": "happy", "urgency": "medium", "has_thread": True},
    # cancellation (4 entries)
    {"id": "email_014", "category": "cancellation", "sentiment": "angry", "urgency": "high", "has_thread": True},
    {"id": "email_015", "category": "cancellation", "sentiment": "neutral", "urgency": "medium", "has_thread": False},
    {"id": "email_016", "category": "cancellation", "sentiment": "confused", "urgency": "low", "has_thread": False},
    {"id": "email_017", "category": "cancellation", "sentiment": "happy", "urgency": "low", "has_thread": True},
    # complaint (3 entries)
    {"id": "email_018", "category": "complaint", "sentiment": "angry", "urgency": "high", "has_thread": False},
    {"id": "email_019", "category": "complaint", "sentiment": "confused", "urgency": "medium", "has_thread": True},
    {"id": "email_020", "category": "complaint", "sentiment": "neutral", "urgency": "low", "has_thread": False},
]


def _build_generation_prompt(spec: dict) -> str:
    """Build prompt to generate a single email scenario with reference reply."""
    thread_instruction = ""
    if spec["has_thread"]:
        thread_instruction = (
            'Include a "prior_thread_summary" field with 1-2 sentences summarizing '
            "a previous exchange between the customer and support team."
        )
    else:
        thread_instruction = 'Set "prior_thread_summary" to null.'

    return f"""Generate a realistic customer support email scenario for a SaaS shared-inbox platform (like Hiver).

Requirements:
- id: "{spec['id']}"
- category: "{spec['category']}"
- sentiment: "{spec['sentiment']}"
- urgency: "{spec['urgency']}"
- {thread_instruction}

The email should feel like a REAL customer wrote it — with natural language, typos are OK, varying formality levels.
For "{spec['sentiment']}" sentiment: {"Express frustration, use strong language (professional but upset)" if spec['sentiment'] == 'angry' else "Ask questions showing lack of understanding" if spec['sentiment'] == 'confused' else "Keep it matter-of-fact and straightforward" if spec['sentiment'] == 'neutral' else "Be positive and appreciative while making a request"}

Also generate:
1. A "reference_reply" — a high-quality customer support agent response (professional, empathetic where needed, actionable, 80-180 words)
2. "reference_reply_key_points" — 3-5 extractable key facts/actions from the reference reply
3. "ideal_reply_traits" — 3-5 abstract qualities a good reply should have

The reference reply should:
- Match tone to the customer's sentiment (apologetic for angry, clear for confused, efficient for neutral, warm for happy)
- Include a specific next step or action
- NOT invent specific dollar amounts, dates, or policy details — use placeholders like "your account" or "our team"
- Be grounded and realistic for a SaaS support context"""


def generate_dataset(output_path: str = None) -> list[EmailScenario]:
    """Generate the full dataset of 20 email scenarios."""
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "dataset.json"
        )

    print("=" * 60)
    print("PHASE 1: DATASET GENERATION")
    print(f"  Generating {len(SCENARIO_SPECS)} email scenarios...")
    print("=" * 60)

    scenarios: list[EmailScenario] = []
    failed: list[str] = []

    for i, spec in enumerate(SCENARIO_SPECS, 1):
        print(f"\n  [{i}/{len(SCENARIO_SPECS)}] Generating {spec['id']} "
              f"({spec['category']}/{spec['sentiment']}/{spec['urgency']})...")
        try:
            prompt = _build_generation_prompt(spec)
            result = call_llm(prompt, response_schema=EmailScenario)
            scenario = EmailScenario.model_validate_json(result)

            # Enforce the spec values (don't trust LLM to echo them correctly)
            scenario.id = spec["id"]
            scenario.category = spec["category"]
            scenario.sentiment = spec["sentiment"]
            scenario.urgency = spec["urgency"]
            if not spec["has_thread"]:
                scenario.prior_thread_summary = None

            scenarios.append(scenario)
            print(f"  [OK] {spec['id']} generated successfully")

        except Exception as e:
            print(f"  [FAIL] {spec['id']} FAILED: {e}")
            failed.append(spec["id"])

    # Save to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [s.model_dump() for s in scenarios],
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n  Dataset saved: {output_path}")
    print(f"  Total generated: {len(scenarios)}/{len(SCENARIO_SPECS)}")
    if failed:
        print(f"  Failed IDs: {failed}")

    return scenarios


if __name__ == "__main__":
    generate_dataset()
