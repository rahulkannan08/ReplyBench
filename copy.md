ReplyBench — Product Requirements Document (v2)
AI Email Reply Generator & Evaluation System for the Hiver Open Challenge

1. Problem Statement
Hiver helps teams run customer email at scale — shared inboxes, right inside Gmail. Their open challenge asks candidates to build:

A dataset of past emails paired with the replies that were sent (the "ground truth")
A Gen-AI reply generator that takes a new incoming email and produces a suggested reply, grounded in the dataset (not just generating in isolation)
An accuracy/evaluation system — the core deliverable — that measures how good each generated response is, with per-response scores and an overall system score
IMPORTANT

The challenge weights clarity of thinking about accuracy/evaluation the heaviest. The evaluator is not an afterthought — it's the main thing being judged.

2. Goals & Success Criteria
Goal	Measurable Criteria
End-to-end runnable	git clone → pip install → add .env → python run.py produces results with zero errors
Dataset with reference replies	20 synthetic email+reply pairs spanning 6 categories, 4 sentiments, 3 urgency levels
Grounded generation	Generator uses few-shot examples retrieved from the dataset — not isolated prompting
Reference-aware evaluation	Scoring compares generated reply against the gold-standard reference, not just abstract quality
Hybrid evaluation	Deterministic checks + LLM-as-judge + reference comparison — multi-signal, not opaque
Metric meta-validation	README defends why each metric reflects real quality, not just a number
Fault tolerance	Any individual item failure is logged and skipped — never crashes the batch
Clean architecture	Small, typed functions; pydantic models; provider logic isolated in one module
3. Tech Stack (Fixed — No Substitutions)
Layer	Choice	Rationale
Language	Python 3.11+	Modern type hints, asyncio support
Primary LLM	Google Gemini API (gemini-2.5-flash) via google-generativeai	Fast, free-tier, structured output support
Fallback LLM	Groq API (llama-3.3-70b-versatile) via groq SDK	Low-latency fallback for reliability under time pressure
Validation	pydantic v2	All LLM structured outputs validated before use
Config	python-dotenv	Keys from .env, never hardcoded
Persistence	Flat JSON files on disk	No DB, no vector store — simplicity
IMPORTANT

Explicitly excluded: LangChain, vector databases, local ML models, any database engine. These add complexity without value for this scope.

4. Architecture Overview
calls
writes
reads dataset forfew-shot retrieval
calls
uses prompts from
reads generated +reference replies
deterministic checks
reference comparison
LLM-as-judge via
rubric from
writes
writes
primary
fallback
run.py(Orchestrator)
Phase 1: Dataset Generation
Phase 2: Reply Generation
Phase 3: Evaluation
/data/generate_dataset.py
/llm/client.pycall_llm()
/data/dataset.json(emails + gold replies)
/generator/generate_reply.py
/generator/prompts.py
/evaluator/evaluate.py
Programmatic Scorers
Key-Point Coverage +Semantic Overlap
/evaluator/rubric.py
/results/per_response_scores.json
/results/aggregate_report.json
Gemini 2.5 Flash
Groq Llama 3.3 70B
5. Repository Structure

ReplyBench/
├── data/
│   ├── __init__.py
│   ├── generate_dataset.py      # Generates 20 synthetic email+reply pairs via LLM
│   ├── schemas.py               # Pydantic models for dataset entries
│   └── dataset.json             # Generated output (gitignored, regenerated on clean run)
├── llm/
│   ├── __init__.py
│   └── client.py                # Single call_llm() — retry + fallback + schema validation
├── generator/
│   ├── __init__.py
│   ├── generate_reply.py        # Few-shot grounded reply generation
│   ├── retriever.py             # Selects few-shot examples from dataset by similarity
│   └── prompts.py               # Reply generation prompt templates
├── evaluator/
│   ├── __init__.py
│   ├── evaluate.py              # Hybrid scoring engine (deterministic + reference + LLM-judge)
│   ├── deterministic.py         # Programmatic scoring functions
│   ├── reference_compare.py     # Reference reply comparison (key-point coverage, overlap)
│   └── rubric.py                # Structured rubric prompt for LLM-as-judge
├── results/
│   ├── per_response_scores.json # Per-response dimension scores + composite
│   └── aggregate_report.json    # Mean/median per dimension, worst category, systemic issues
├── run.py                       # Single-command full pipeline orchestrator
├── README.md                    # Approach, metrics defense, run instructions
├── .env.example                 # Template for API keys
├── .gitignore                   # .env, __pycache__, results/, data/dataset.json
└── requirements.txt             # google-generativeai, groq, pydantic, python-dotenv
6. Detailed Component Specifications
6.1 Shared LLM Client — /llm/client.py
IMPORTANT

Build this first. Every downstream module depends on it. Test with a real API call to both providers before moving on.

Function Signature
python

def call_llm(
    prompt: str,
    response_schema: type[BaseModel] | None = None,
    max_retries: int = 2
) -> str:
Behavior
Step	Detail
1. Primary call	Send to Gemini (gemini-2.5-flash)
2. Retry on failure	On any exception (rate limit, timeout, 5xx), retry up to max_retries times with exponential backoff (base 1.5s)
3. Fallback	If Gemini exhausts retries, fall back once to Groq (llama-3.3-70b-versatile)
4. Structured output	If response_schema provided: instruct model to return JSON matching schema; validate with pydantic; on parse failure, retry once with a stricter "return ONLY valid JSON, no markdown fences" suffix
5. Logging	Print which provider served each call (provider name, latency, success/failure) to stdout
6. Total failure	Raise a clear exception — the caller decides whether to skip or halt
Constraints
Read GEMINI_API_KEY and GROQ_API_KEY from environment via dotenv
No global mutable state, no singleton clients — clean, testable function
No uncaught exceptions — always retry/fallback before surfacing error
6.2 Dataset — /data/
What's Different from v1
The official challenge says: "Create a dataset of past emails paired with the replies that were sent." This means each entry must include a gold-standard reference reply — not just ideal_reply_traits. The reference reply is what a human agent actually sent (or, in our synthetic case, what a high-quality agent would send).

Pydantic Schema — EmailScenario (in /data/schemas.py)
python

class EmailScenario(BaseModel):
    id: str                              # e.g., "email_001"
    subject: str                         # Email subject line
    customer_email_body: str             # The customer's actual email text
    category: str                        # billing | bug_report | refund_request | feature_question | cancellation | complaint
    sentiment: str                       # angry | neutral | confused | happy
    urgency: str                         # low | medium | high
    prior_thread_summary: str | None     # Shared-inbox context (null for cold emails)
    reference_reply: str                 # ★ Gold-standard reply (what a human agent sent)
    reference_reply_key_points: list[str]  # ★ Extractable key points from the reference reply
    ideal_reply_traits: list[str]        # What a good reply should include (abstract)
NOTE

Why both reference_reply and ideal_reply_traits?

reference_reply is the concrete gold-standard text — used for semantic comparison scoring
reference_reply_key_points are the factual/action points from the reference — used for key-point coverage scoring
ideal_reply_traits are abstract quality expectations — used for LLM-as-judge rubric context
Generation Strategy
The dataset is generated in two LLM calls per entry:

Call 1: Generate the customer email with metadata (category, sentiment, urgency, thread context)
Call 2: Generate the reference reply + key points for that email (separate call to avoid the model "knowing" its own answer)
Alternatively, generate all in one call but with explicit instructions to make the reference reply realistic and grounded.

Variation Requirements
Dimension	Values	Distribution
Category	billing, bug_report, refund_request, feature_question, cancellation, complaint	At least 3 entries per category
Sentiment	angry, neutral, confused, happy	Mixed across categories
Urgency	low, medium, high	Roughly balanced
Prior thread	Present / null	~40% should have prior thread context
README Documentation
Explain:

Why synthetic (no public customer email corpus exists with paired replies)
How diversity was enforced across categories/sentiments
Show a sample entry
Acknowledge limitation: synthetic data may not capture real-world edge cases
6.3 Generator — /generator/
What's Different from v1
The challenge says: "Ground the generation in your dataset." This means the generator should use the dataset as context — not just generate replies in isolation. We'll use few-shot retrieval: for each incoming email, find the most similar past emails from the dataset and include their reference replies as examples in the prompt.

Few-Shot Retrieval Strategy (/generator/retriever.py)
python

def retrieve_few_shot_examples(
    target_email: EmailScenario,
    dataset: list[EmailScenario],
    k: int = 3
) -> list[EmailScenario]:
    """
    Select k most similar emails from the dataset to use as few-shot examples.
    
    Similarity heuristic (no vector DB, no embeddings model):
    1. Category match (highest priority — same category emails are most relevant)
    2. Sentiment match (secondary — similar tone context)
    3. Urgency match (tertiary)
    
    Excludes the target email itself from candidates.
    Returns up to k examples, prioritizing those with prior_thread_summary
    if the target also has one.
    """
NOTE

Why not embeddings/vector DB? The challenge excludes vector databases and local ML models. A category+sentiment+urgency heuristic is simple, transparent, defensible, and appropriate for a 20-item dataset where full semantic search is overkill.

Pydantic Schema — GeneratedReply
python

class GeneratedReply(BaseModel):
    id: str                        # Matches the EmailScenario id
    reply_subject: str             # Reply email subject
    reply_body: str                # The generated reply text
    tone_used: str                 # e.g., "empathetic", "professional", "concise"
    suggested_next_action: str     # What should happen next
    confidence: float              # 0.0 – 1.0
Prompt Design (/generator/prompts.py)
The prompt template includes:

System instruction: You are a customer support agent for a SaaS company. Respond helpfully and professionally.
Few-shot examples (2-3): Past emails + their actual replies from the dataset
Target email: All context fields (category, sentiment, urgency, prior thread)
Tone instruction: Empathetic for angry/confused, concise for neutral/simple, warm-professional for happy
Grounding guard: "Do NOT invent policy details, refund amounts, or promises not grounded in the input context or the examples provided"
Concurrency
ThreadPoolExecutor with 4–6 workers
Each item wrapped in try/except — failed IDs logged, batch continues
Respect reasonable RPM limits for both providers
6.4 Evaluator — /evaluator/ — ★ CORE DELIVERABLE
CAUTION

This is what Hiver cares about most. The evaluation system must answer: "What does 'accurate' even mean for a suggested reply?" and defend that answer rigorously.

Our Answer to "What Does Accurate Mean?"
Exact match is too strict — two very different replies can both be excellent. Instead, accuracy for email replies is multi-dimensional:

Signal	Question It Answers	Why It Matters
Reference similarity	Does the generated reply convey the same meaning as the gold standard?	Validates the generator is learning from the dataset, not hallucinating
Key-point coverage	Does the reply hit the same factual/action points as the reference?	A reply can be phrased differently but must address the same substantive points
Relevance	Does it address what the customer actually asked?	Off-topic replies are useless regardless of quality
Faithfulness	Does it avoid hallucinated facts/policies/promises?	Customer support errors have real business/legal consequences
Tone match	Is the tone appropriate for the customer's emotional state?	Angry customers getting a chipper response = escalation
Completeness	Does it provide a concrete next step?	"Thank you for reaching out" alone is not helpful
Conciseness	Is it appropriately brief?	Customers don't read novels
Safety	No overpromising, no fabricated PII, no unsafe claims?	Compliance requirement
Three Scoring Tiers
Tier 3: LLM-as-Judge
Relevance
Faithfulness
Safety Compliance
Tier 2: Deterministic Checks
Conciseness(word count)
Tone Match(lexicon analysis)
Completeness(action phrase detection)
Tier 1: Reference Comparison
Key-Point Coverage(deterministic + LLM)
Semantic Overlap(LLM-based comparison)
Composite Score
Dimension Details
Tier 1: Reference Comparison (new in v2)
Dimension	Range	Method	Implementation
key_point_coverage	0–10	LLM-assisted	Extract key points from reference reply; ask LLM "which of these key points are addressed in the generated reply?"; score = (points covered / total points) × 10
semantic_similarity	0–10	LLM-as-judge	Single LLM call: "On a scale of 0-10, how semantically similar is Reply A to Reply B? Consider meaning, not exact wording." With justification.
Tier 2: Deterministic Checks (no LLM needed)
Dimension	Range	Implementation
conciseness	0–10	Word-count against a reasonable range (50–200 words). Score 10 if within ideal, degrade linearly outside
tone_appropriateness	0–10	Keyword/lexicon sentiment matching: apologetic words for angry customers, solution-focused for confused, etc.
completeness	0–10	Presence-check for action phrases: "next step", "I'll", "we will", "follow up", scheduling language
Tier 3: LLM-as-Judge (subjective quality)
Dimension	Range	Implementation
relevance	0–10	Does the reply address what the customer actually asked?
correctness_faithfulness	0–10	No hallucinated facts/policy/promises not in the input
safety_compliance	0–10	No overpromising, no unsafe claims, no fabricated PII
NOTE

Tier 3 uses a single batched LLM call per response — one structured prompt covers relevance + faithfulness + safety. This conserves API calls while keeping scores independent.

Composite Score & Weights
Dimension	Weight	Tier	Justification
correctness_faithfulness	0.20	LLM-judge	Factual errors carry real business/legal risk
safety_compliance	0.15	LLM-judge	Overpromising = liability
key_point_coverage	0.20	Reference	Core "accuracy" signal — did we hit the right substance?
semantic_similarity	0.10	Reference	Validates overall alignment with gold standard
relevance	0.15	LLM-judge	Must answer what was asked
completeness	0.10	Deterministic	Customers need actionable next steps
tone_appropriateness	0.05	Deterministic	Matters, but less catastrophic if slightly off
conciseness	0.05	Deterministic	Soft preference only
Formula: composite = Σ(weight_i × score_i) → range 0–10

Why This Metric Is Right (README defense)
Why not exact match? Two replies can say the same thing very differently. Exact match would penalize valid paraphrasing and stylistic variation.
Why hybrid? A single "rate 1-10" LLM call is opaque and unrepeatable. Deterministic checks are transparent and reproducible. LLM-as-judge handles what rules can't (subjective relevance, hallucination detection). Reference comparison grounds everything in the dataset.
Why these weights? Weighted by business impact: a factually wrong or unsafe reply does more damage than a verbose one. Key-point coverage is weighted high because it's the most objective "accuracy" signal.
How we validate the metric reflects real quality: We include per-dimension justifications in every score so a human can audit whether the score matches their judgment. Systemic analysis in the aggregate report surfaces patterns (e.g., "tone scores consistently low for angry emails") that confirm or challenge the metric's validity.
Per-Response Output Schema
python

class DimensionScore(BaseModel):
    value: float              # 0-10
    method: str               # "llm_judge" | "deterministic" | "reference_compare"
    justification: str = ""   # For LLM-judged dimensions
    detail: str = ""          # For deterministic dimensions
class ResponseEvaluation(BaseModel):
    id: str
    scores: dict[str, DimensionScore]   # 8 dimensions
    composite_score: float
    flags: list[str]                    # e.g., ["low_key_point_coverage", "hallucination_risk"]
Aggregate Report Schema
python

class DimensionStats(BaseModel):
    mean: float
    median: float
    min: float
    max: float
    std_dev: float
class AggregateReport(BaseModel):
    dimension_stats: dict[str, DimensionStats]   # Stats per dimension
    overall_composite_mean: float
    overall_composite_median: float
    worst_performing_category: str
    best_performing_category: str
    systemic_issues: list[str]                   # e.g., "tone mismatch in angry-sentiment entries"
    total_evaluated: int
    total_failed: int
6.5 Pipeline Orchestrator — run.py
No
Yes
Start
dataset.json\nexists?
Generate Dataset\n(20 email+reply pairs)
Load Dataset
Generate Replies\n(few-shot grounded,\nconcurrent)
Evaluate All Replies\n(3-tier hybrid scoring,\nconcurrent)
Write /results/\nper_response + aggregate
Print Summary Table\nto stdout
Single command: python run.py
Idempotent dataset: Skip generation if dataset.json already exists
Fault-tolerant batches: Both reply generation and evaluation tolerate individual failures
Stdout summary: Formatted table with per-dimension averages + overall composite + worst category
6.6 README.md Structure
Section	Content
1. Approach	Why generator and evaluator are separate systems; what "accuracy" means for email replies
2. Tech Stack	Gemini primary / Groq fallback — reliability rationale
3. Dataset	How built, why synthetic, schema rationale, sample entry
4. Generator	Few-shot grounding strategy, what context is passed, tone logic
5. Why This Evaluation Metric Is Right	Defense of 8 dimensions, 3 tiers, weights; why hybrid + reference comparison beats a single LLM call; how we validate the metric reflects real quality
6. Known Limitations	Honest constraints (synthetic data, no real user feedback loop, lexicon-based tone is simplistic)
7. How to Run	Exact commands from git clone to seeing results — zero assumed local state
8. AI Tools Used	How AI tools were used during development (required by challenge)
7. Engineering Constraints (Non-Negotiable)
Constraint	Detail
No hardcoded keys	.env + .env.example checked in; .env in .gitignore
No hardcoded paths	All file paths relative to project root
Fault tolerance	Every batch operation tolerates individual item failures without crashing
Clean functions	Small, clearly named, typed signatures
Provider isolation	Fallback logic lives ONLY in /llm/client.py — no other module knows or cares which provider served a call
End-to-end verification	Simulate a clean clone and run README instructions to confirm zero implicit local state
8. Build Order (Strict Sequential, Verify Each Phase)
Phase	Module	Verification
Phase 0	Project scaffold + .gitignore + requirements.txt + .env.example	Files exist, git initialized
Phase 1	/llm/client.py	Test with a hello-world call to both Gemini and Groq; confirm logging
Phase 2	/data/schemas.py + /data/generate_dataset.py	Generate and inspect dataset.json — verify schema, variety, reference replies
Phase 3	/generator/retriever.py + /generator/prompts.py + /generator/generate_reply.py	Test on 2–3 entries; inspect reply quality, verify few-shot examples appear in prompt
Phase 4	/evaluator/deterministic.py + /evaluator/reference_compare.py + /evaluator/rubric.py + /evaluator/evaluate.py	Test on 2–3 entries; inspect all 8 dimension scores
Phase 5	run.py	Wire full pipeline; run end-to-end
Phase 6	README.md	Complete documentation; clean-clone test
Open Questions
IMPORTANT

Please clarify the following before we begin development:

API Keys — Do you have both GEMINI_API_KEY and GROQ_API_KEY ready?

Git initialization — Should I initialize a git repo + .gitignore as part of Phase 0, or will you handle that?

Dataset size — I'm recommending 20 entries (enough variety for 6 categories × reasonable coverage, without excessive API calls). Acceptable?

Concurrency model — I recommend ThreadPoolExecutor (simpler than async, avoids coloring the whole codebase). OK?

Evaluation scope: 8 vs 6 dimensions — The official challenge doesn't prescribe specific dimensions. v2 adds key_point_coverage and semantic_similarity (reference comparison) for a total of 8 dimensions across 3 tiers. This is more complex but directly addresses "what does accurate mean?" Do you want to keep all 8, or simplify?

Stdout verbosity — I recommend moderate: phase headers + per-item progress + provider logs + final summary table. OK?

Verification Plan
Automated Tests
bash

# Phase 1: LLM client smoke test
python -c "from llm.client import call_llm; print(call_llm('Say hello in one word'))"
# Phase 2: Dataset generation + schema validation
python data/generate_dataset.py
python -c "import json; from data.schemas import EmailScenario; [EmailScenario(**e) for e in json.load(open('data/dataset.json'))]; print('All valid')"
# Phase 3: Generator test (2-3 entries)
python -c "from generator.generate_reply import generate_replies; generate_replies(limit=3)"
# Phase 4: Evaluator test
python -c "from evaluator.evaluate import evaluate_replies; evaluate_replies(limit=3)"
# Full pipeline
python run.py
Manual Verification
Inspect data/dataset.json for realistic variety + quality reference replies
Inspect generated replies for few-shot grounding (tone alignment, no hallucinated policies)
Inspect results/per_response_scores.json for reasonable 8-dimension score distributions
Inspect results/aggregate_report.json for meaningful systemic insights
Clean-clone test: fresh directory, only .env, run README instructions end-to-end