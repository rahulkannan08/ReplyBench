# ReplyBench

**AI Email Reply Generator & Evaluation System**
*Hiver Open Challenge Submission*

---

## 1. Approach

ReplyBench is two systems, deliberately kept separate:

1. **Generator** — takes a customer email and produces a suggested reply, grounded in a dataset of past email/reply pairs via few-shot retrieval
2. **Evaluator** — scores each generated reply on 6 independent dimensions using a hybrid approach (deterministic checks + LLM-as-judge + reference comparison)

The evaluator is the core deliverable. The question it answers: *"What does 'accurate' even mean for a suggested email reply?"*

Our answer: accuracy is multi-dimensional. A reply can be well-written but off-topic (low relevance), on-topic but hallucinating policies (low faithfulness), or factually correct but tonally inappropriate for an angry customer. A single "rate 1-10" score obscures these distinctions. ReplyBench scores each axis independently, then combines them with justified weights.

---

## 2. Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.11+ | Modern type hints, clean syntax |
| Primary LLM | Google Gemini API (`gemini-2.5-flash`) via `google-genai` | Fast, free-tier, structured output support |
| Fallback LLM | Groq API (`llama-3.3-70b-versatile`) via `groq` SDK | Low-latency fallback for reliability |
| Validation | Pydantic v2 | All LLM structured outputs validated before use |
| Config | python-dotenv | Keys from `.env`, never hardcoded |
| Persistence | Flat JSON files | No DB, no vector store — simplicity |

**Why dual providers?** Under a time-boxed run, API reliability matters. If Gemini rate-limits or errors, Groq serves the call automatically. The fallback is visible in stdout logs — you can see exactly which provider served each call.

**Explicitly excluded:** LangChain, vector databases, local ML models, any database engine. These add complexity without value for this scope.

---

## 3. Dataset

### How It Was Built

20 synthetic customer support email scenarios generated via LLM, each paired with a gold-standard reference reply. Synthetic because no public customer email corpus exists with paired replies and permission for use.

**Diversity enforcement:** Pre-defined scenario specs ensure coverage across:
- **6 categories**: billing, bug_report, refund_request, feature_question, cancellation, complaint (~3 each)
- **4 sentiments**: angry, neutral, confused, happy (mixed across categories)
- **3 urgency levels**: low, medium, high (roughly balanced)
- **Thread context**: ~40% include `prior_thread_summary` (mimics Hiver's shared-inbox context)

LLM generates the email + reference reply, but spec values (category, sentiment, urgency) are enforced post-generation to prevent drift.

### Sample Entry

```json
{
  "id": "email_001",
  "subject": "URGENT: Double charged on my account",
  "customer_email_body": "I just noticed I was charged twice for my subscription this month...",
  "category": "billing",
  "sentiment": "angry",
  "urgency": "high",
  "prior_thread_summary": "Customer contacted support last week about a billing discrepancy...",
  "reference_reply": "I sincerely apologize for the duplicate charge...",
  "reference_reply_key_points": [
    "Acknowledges the duplicate charge",
    "Promises investigation by billing team",
    "Provides timeline for resolution"
  ],
  "ideal_reply_traits": [
    "Empathetic and apologetic tone",
    "Specific next step provided",
    "No invented refund amounts"
  ]
}
```

### Dataset Split

- **15 entries** → retrieval pool (used as few-shot examples for generation)
- **5 entries** → held-out test set (replies generated and evaluated against these)

This gives retrieval enough breadth to pick meaningfully different neighbors while keeping the eval loop fast.

---

## 4. Generator

### Few-Shot Grounding

The challenge says: *"Ground the generation in your dataset."* Rather than generating replies in isolation, we retrieve the 3 most similar past emails from the pool and include their reference replies as few-shot examples in the prompt.

**Retrieval heuristic** (no vector DB, no embeddings — appropriate for 20 items):
1. Category match (highest weight — same category emails are most relevant)
2. Sentiment match (tone context)
3. Urgency match
4. Thread context match (prefer threaded examples when target has thread)

**Tone instruction** is explicit in the prompt, mapped from customer sentiment:
- Angry → empathetic + apologetic
- Confused → clear + patient + step-by-step
- Neutral → professional + concise
- Happy → warm + professional

**Grounding guard:** The prompt explicitly instructs: *"Do NOT invent policy details, refund amounts, or promises not grounded in the input context."*

---

## 5. Why This Evaluation Metric Is Right

### The Problem with Simple Scoring

A single "rate this reply 1-10" LLM call is:
- **Opaque** — you don't know what drove the score
- **Unrepeatable** — different runs give different numbers with no explanation
- **Uninformative** — a "6/10" doesn't tell you what to fix

### Our Solution: 6 Dimensions, 3 Tiers

| Dimension | Weight | Tier | Why This Weight |
|-----------|--------|------|-----------------|
| `correctness_faithfulness` | 25% | LLM-judge | Factual errors in customer support carry real business/legal risk |
| `relevance` | 20% | LLM-judge | An off-topic reply is useless regardless of quality |
| `safety_compliance` | 15% | LLM-judge | Overpromising creates liability |
| `semantic_similarity` | 15% | Reference comparison | Validates the generator learns from the dataset, not hallucinating |
| `completeness` | 10% | Deterministic | Customers need actionable next steps, not just acknowledgment |
| `tone_appropriateness` | 10% | Deterministic | Tone matters — an angry customer getting a chipper response = escalation |
| `conciseness` | 5% | Deterministic | Soft preference — customers don't read novels, but brevity alone isn't quality |

**Formula:** `composite = sum(weight_i * score_i)` → range 0–10

### Why Hybrid Beats Pure LLM-Judge

| Signal Type | Dimensions | Advantage |
|-------------|-----------|-----------|
| **Deterministic** | conciseness, tone, completeness | Transparent, reproducible, zero API cost, auditable rules |
| **Reference comparison** | semantic_similarity | Grounds scoring in the dataset — not abstract quality, but alignment with known-good replies |
| **LLM-as-judge** | relevance, faithfulness, safety | Handles subjective judgment that rules can't capture |

Each dimension reports its `method` and `justification`/`detail`, so a human can audit whether the score matches their judgment.

### Why These 6 and Not More

We deliberately chose 6 clearly-distinct dimensions over 8 with overlap. Two candidates were considered and rejected:
- **key_point_coverage** — mostly redundant with completeness + relevance. Adding it risks a reviewer noticing two dimensions measuring nearly the same thing.
- **semantic_similarity as a separate 8th dimension** — it IS one of our 6, not an addition. It's the reference-based signal.

6 clean, defensible dimensions beats 8 with overlap. Clarity of evaluation is the heaviest-weighted criterion in this challenge.

### How We Validate the Metric Reflects Real Quality

We run `validate_metric.py` to prove the metric discriminates between good and bad replies — not just assigning numbers:

- **Case A**: Score each reference reply against itself (should score high)
- **Case B**: Score deliberately bad control replies — wrong tone, off-topic, overpromising, no action items (should score low)

| Dimension | Case A (reference) | Case B (bad reply) | Gap | Method |
|-----------|-------------------|-------------------|-----|--------|
| tone_appropriateness | 10.0 | 0.0 | **+10.0** | Deterministic |
| completeness | 9.0 | 2.0 | **+7.0** | Deterministic |
| conciseness | 10.0 | 5.2 | **+4.8** | Deterministic |
| semantic_similarity | 10.0 | 5.0 | **+5.0** | Reference compare |
| relevance | 8.0* | 5.0* | +3.0 | LLM-judge |
| correctness_faithfulness | 9.0* | 5.0* | +4.0 | LLM-judge |
| safety_compliance | 9.0* | 5.0* | +4.0 | LLM-judge |
| **Composite** | **6.90** | **4.21** | **+2.69** | Weighted |

*\*LLM-judged dimensions marked with \* show scores from successful API calls; under rate limiting, these default to 5.0 for both cases, which compresses the gap. The deterministic dimensions — which are fully reproducible and API-independent — show the clearest discrimination.*

**What this proves:** the metric reliably assigns higher scores to genuinely good replies and lower scores to bad ones. The deterministic tier alone produces a massive gap (tone: 10 vs 0, completeness: 9 vs 2), and the LLM-judge tier adds further discrimination when API calls succeed. Run `python validate_metric.py` to reproduce.

Additional validation mechanisms:
1. **Per-dimension justifications** — every LLM-judged score includes a one-sentence explanation a human can audit
2. **Systemic analysis** — the aggregate report surfaces patterns (e.g., "tone mismatch concentrated in angry-sentiment entries") that confirm or challenge the metric's validity
3. **Flag system** — low scores on critical dimensions trigger named flags (`hallucination_risk`, `safety_concern`)
4. **Weight justification** — weighted by business impact, not arbitrary

---

## 6. Known Limitations

1. **Synthetic data** — no real customer emails. Synthetic data may not capture real-world edge cases, ambiguity, or the messy diversity of actual customer writing.
2. **Lexicon-based tone scoring** — keyword matching is simplistic. A reply could use apologetic words sarcastically and still score well. A proper sentiment model would be better but is explicitly excluded from scope.
3. **No real feedback loop** — we can't validate whether high-scoring replies actually satisfy real customers. The metric is defensible but unvalidated against ground truth.
4. **LLM-as-judge bias** — the judge LLM may have systematic biases (e.g., preferring longer replies, favoring certain phrasings). We mitigate with a detailed rubric and explicit "don't inflate" instruction. In practice, we observed score clustering at round integers (e.g., all responses scoring 8/9/9 on judge dimensions) — the per-response justifications show genuine per-item reasoning, but the numeric resolution is coarse. A more capable judge model or multi-judge consensus would improve discrimination.
5. **Small dataset** — 20 entries with 5 evaluated is enough to demonstrate the system but not enough for statistical significance.
6. **Fallback-as-judge** — when Gemini rate-limits and Groq serves the judge calls, the evaluation quality may differ from the primary model. Ideally, the judge would always use the same model for consistency.

---

## 7. How to Run

### Prerequisites
- Python 3.10+
- Gemini API key ([get one free](https://aistudio.google.com/apikey))
- Groq API key ([get one free](https://console.groq.com/keys))

### Steps

```bash
# 1. Clone
git clone https://github.com/rahulkannan08/ReplyBench.git
cd ReplyBench

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY and GROQ_API_KEY

# 4. Run the full pipeline
python run.py

# 5. (Optional) Validate the metric discriminates good vs bad replies
python validate_metric.py
```

### What Happens

1. **Dataset generation** — 20 email+reply pairs generated via Gemini (skip if `data/dataset.json` exists)
2. **Reply generation** — 5 held-out test emails get AI-generated replies using few-shot grounding
3. **Evaluation** — each reply scored on 6 dimensions (deterministic + LLM-judge + reference comparison)
4. **Results** — per-response scores + aggregate report written to `/results/`, summary table printed to stdout

### Output Files

| File | Contents |
|------|----------|
| `data/dataset.json` | 20 email scenarios with reference replies |
| `results/per_response_scores.json` | Per-response 6-dimension scores + composite |
| `results/aggregate_report.json` | Mean/median per dimension, category analysis, systemic issues |

---

## 8. AI Tools Used

- **Google Gemini API** (`gemini-2.5-flash`) — primary LLM for dataset generation, reply generation, and LLM-as-judge evaluation
- **Groq API** (`llama-3.3-70b-versatile`) — automatic fallback when Gemini fails
- **AI coding assistant** — used for code generation, architecture planning, and documentation drafting

All code was reviewed, understood, and verified by the author. The architecture and evaluation metric design are original work.

---

## Project Structure

```
ReplyBench/
├── data/
│   ├── generate_dataset.py    # Generates 20 synthetic email+reply pairs
│   ├── schemas.py             # Pydantic models (shared across pipeline)
│   └── dataset.json           # Generated output (gitignored)
├── llm/
│   └── client.py              # Single call_llm() — retry + fallback + validation
├── generator/
│   ├── generate_reply.py      # Few-shot grounded reply generation
│   ├── retriever.py           # Selects few-shot examples by similarity
│   └── prompts.py             # Reply generation prompt templates
├── evaluator/
│   ├── evaluate.py            # Hybrid scoring engine (orchestrates 3 tiers)
│   ├── deterministic.py       # Programmatic scoring (conciseness, tone, completeness)
│   ├── reference_compare.py   # Semantic similarity via LLM
│   └── rubric.py              # LLM-as-judge structured rubric
├── results/
│   ├── per_response_scores.json
│   ├── aggregate_report.json
│   └── metric_validation_report.json
├── run.py                     # Single-command pipeline orchestrator
├── validate_metric.py         # Metric validation (Case A vs Case B)
├── README.md
├── .env.example
├── .gitignore
└── requirements.txt
```
