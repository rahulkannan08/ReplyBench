"""
ReplyBench -- Single-command full pipeline orchestrator.

Usage: python run.py

Phases:
  1. Generate dataset (skip if data/dataset.json exists)
  2. Split dataset: 15 retrieval pool + 5 held-out test
  3. Generate replies for 5 test entries (concurrent, fault-tolerant)
  4. Evaluate all replies (3-tier hybrid scoring, concurrent)
  5. Write results to /results/
  6. Print summary table to stdout
"""

import json
import os
import sys
import time

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.schemas import EmailScenario, GeneratedReply
from data.generate_dataset import generate_dataset
from generator.generate_reply import generate_replies
from evaluator.evaluate import evaluate_replies, build_aggregate_report, WEIGHTS


DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PER_RESPONSE_PATH = os.path.join(RESULTS_DIR, "per_response_scores.json")
AGGREGATE_PATH = os.path.join(RESULTS_DIR, "aggregate_report.json")

# Fixed split: first 15 = retrieval pool, last 5 = held-out test
POOL_SIZE = 15
TEST_SIZE = 5


def _load_dataset() -> list[EmailScenario]:
    """Load dataset from JSON file."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EmailScenario.model_validate(entry) for entry in data]


def _print_summary_table(evaluations, aggregate, scenarios):
    """Print a formatted summary table to stdout."""
    scenario_map = {s.id: s for s in scenarios}

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Per-dimension averages
    print("\n--- Dimension Averages (weight) ---")
    print(f"  {'Dimension':<28} {'Mean':>6} {'Median':>7} {'Min':>5} {'Max':>5} {'Weight':>7}")
    print("  " + "-" * 64)
    for dim in WEIGHTS:
        if dim in aggregate.dimension_stats:
            stats = aggregate.dimension_stats[dim]
            weight = WEIGHTS[dim]
            print(
                f"  {dim:<28} {stats.mean:>6.1f} {stats.median:>7.1f} "
                f"{stats.min:>5.1f} {stats.max:>5.1f} {weight:>7.0%}"
            )

    # Overall composite
    print(f"\n  Overall Composite Mean:   {aggregate.overall_composite_mean:.2f} / 10.00")
    print(f"  Overall Composite Median: {aggregate.overall_composite_median:.2f} / 10.00")

    # Per-response breakdown
    print("\n--- Per-Response Scores ---")
    print(f"  {'ID':<12} {'Category':<18} {'Composite':>9} {'Flags'}")
    print("  " + "-" * 64)
    for ev in evaluations:
        scenario = scenario_map.get(ev.id)
        cat = scenario.category if scenario else "unknown"
        flags_str = ", ".join(ev.flags) if ev.flags else "none"
        print(f"  {ev.id:<12} {cat:<18} {ev.composite_score:>9.2f} {flags_str}")

    # Category breakdown
    print(f"\n  Best category:  {aggregate.best_performing_category}")
    print(f"  Worst category: {aggregate.worst_performing_category}")

    # Systemic issues
    print("\n--- Systemic Issues ---")
    for issue in aggregate.systemic_issues:
        print(f"  - {issue}")

    # Stats
    print(f"\n  Total evaluated: {aggregate.total_evaluated}")
    print(f"  Total failed:    {aggregate.total_failed}")
    print("=" * 80)


def main():
    """Run the full ReplyBench pipeline."""
    start_time = time.time()

    print("\n" + "#" * 80)
    print("#  ReplyBench -- AI Email Reply Generator & Evaluation System")
    print("#  Hiver Open Challenge Submission")
    print("#" * 80)

    # --- Phase 1: Dataset ---
    if os.path.exists(DATASET_PATH):
        print(f"\n[SKIP] Dataset already exists at {DATASET_PATH}")
        print("  Delete data/dataset.json and re-run to regenerate.")
        scenarios = _load_dataset()
        print(f"  Loaded {len(scenarios)} scenarios from disk.")
    else:
        scenarios = generate_dataset(DATASET_PATH)

    if len(scenarios) < POOL_SIZE + TEST_SIZE:
        print(
            f"\n[WARN] Only {len(scenarios)} scenarios generated. "
            f"Need {POOL_SIZE + TEST_SIZE}. Using all as pool, last {min(TEST_SIZE, len(scenarios))} as test."
        )
        pool_entries = scenarios[: max(1, len(scenarios) - TEST_SIZE)]
        test_entries = scenarios[max(1, len(scenarios) - TEST_SIZE) :]
    else:
        pool_entries = scenarios[:POOL_SIZE]
        test_entries = scenarios[POOL_SIZE : POOL_SIZE + TEST_SIZE]

    print(f"\n  Dataset split: {len(pool_entries)} pool + {len(test_entries)} test")

    # --- Phase 2: Reply Generation ---
    replies = generate_replies(pool_entries, test_entries, max_workers=5)

    if not replies:
        print("\n[ERROR] No replies generated. Cannot evaluate. Exiting.")
        sys.exit(1)

    # --- Phase 3: Evaluation ---
    evaluations = evaluate_replies(scenarios, replies, max_workers=3)

    if not evaluations:
        print("\n[ERROR] No evaluations completed. Exiting.")
        sys.exit(1)

    # --- Phase 4: Write Results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Per-response scores
    per_response_data = [ev.model_dump() for ev in evaluations]
    with open(PER_RESPONSE_PATH, "w", encoding="utf-8") as f:
        json.dump(per_response_data, f, indent=2, ensure_ascii=False)

    # Aggregate report
    aggregate = build_aggregate_report(evaluations, scenarios)
    with open(AGGREGATE_PATH, "w", encoding="utf-8") as f:
        json.dump(aggregate.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"\n  Results written to:")
    print(f"    {PER_RESPONSE_PATH}")
    print(f"    {AGGREGATE_PATH}")

    # --- Phase 5: Summary ---
    _print_summary_table(evaluations, aggregate, scenarios)

    elapsed = time.time() - start_time
    print(f"\n  Total pipeline time: {elapsed:.1f}s")
    print(f"\n  Done. See /results/ for full output.\n")


if __name__ == "__main__":
    main()
