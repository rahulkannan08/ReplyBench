"""
Concurrent reply generation for held-out test entries.

Uses few-shot examples from the retrieval pool (15 entries) to ground
replies for the 5 held-out test entries. ThreadPoolExecutor for concurrency.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.schemas import EmailScenario, GeneratedReply
from generator.retriever import retrieve_few_shot_examples
from generator.prompts import build_reply_prompt
from llm.client import call_llm


def _generate_single_reply(
    target: EmailScenario,
    pool: list[EmailScenario],
) -> GeneratedReply:
    """Generate a reply for a single email using few-shot grounding."""
    # Retrieve similar examples from the pool
    examples = retrieve_few_shot_examples(target, pool, k=3)

    # Build the grounded prompt
    prompt = build_reply_prompt(target, examples)

    # Call LLM with structured output
    result = call_llm(prompt, response_schema=GeneratedReply)
    reply = GeneratedReply.model_validate_json(result)

    # Enforce the correct ID
    reply.id = target.id

    return reply


def generate_replies(
    pool_entries: list[EmailScenario],
    test_entries: list[EmailScenario],
    max_workers: int = 5,
) -> list[GeneratedReply]:
    """
    Generate replies for all test entries using few-shot grounding from pool.

    Args:
        pool_entries: 15-entry retrieval pool for few-shot examples.
        test_entries: 5 held-out entries to generate replies for.
        max_workers: ThreadPoolExecutor worker count.

    Returns:
        List of successfully generated replies.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: REPLY GENERATION")
    print(f"  Generating replies for {len(test_entries)} test entries")
    print(f"  Using {len(pool_entries)} entries as retrieval pool")
    print(f"  Workers: {max_workers}")
    print("=" * 60)

    replies: list[GeneratedReply] = []
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(_generate_single_reply, entry, pool_entries): entry.id
            for entry in test_entries
        }

        for i, future in enumerate(as_completed(future_to_id), 1):
            entry_id = future_to_id[future]
            try:
                reply = future.result()
                replies.append(reply)
                print(f"\n  [{i}/{len(test_entries)}] {entry_id} -- generated "
                      f"({len(reply.reply_body.split())} words, "
                      f"tone={reply.tone_used}, conf={reply.confidence:.2f})")
            except Exception as e:
                failed.append(entry_id)
                print(f"\n  [{i}/{len(test_entries)}] {entry_id} -- FAILED: {e}")

    # Sort by ID for consistent output
    replies.sort(key=lambda r: r.id)

    print(f"\n  Replies generated: {len(replies)}/{len(test_entries)}")
    if failed:
        print(f"  Failed IDs: {failed}")

    return replies
