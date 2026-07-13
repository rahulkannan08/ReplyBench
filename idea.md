test project (Hiver — Open Challenge
100 minutes. One repo. Build AI that writes better email.

Hiver helps teams run customer email at scale — shared inboxes, right inside Gmail. We're growing the Hiver team, and instead of a resume screen we run an open challenge: start when you're ready, get 100 minutes, and build something real. The task is deliberately close to what we care about — using generative AI to suggest great email replies, and knowing how good those replies actually are.

Rules
The clock starts the moment you click Start, and runs for 100 minutes.
Work in a public GitHub repository — you'll share the link when you submit.
Use any language, libraries, and AI tools you like. Tell us how in your README.
All work must be your own.
Submit before the timer hits zero. Shipping something that runs beats a polished idea that doesn't.
You'll submit
A public GitHub repository URL.
The dataset (or a script that generates/fetches it) and how you built it.
The Gen-AI response generator, runnable end-to-end.
The accuracy/evaluation system, with per-response and overall scores.
A README: your approach, why your accuracy metric is right, and how to run it.)

You are building a submission for a 100-minute AI engineering challenge from Hiver 
(a shared-inbox/customer email platform). The deliverable is a public GitHub repo 
containing two systems: (1) a Gen-AI email reply generator, and (2) an evaluation 
system that scores reply quality per-response and in aggregate. Build this fully 
end-to-end, runnable with one command, with no placeholder/TODO code left in. 
This must be production-quality in structure — clean modules, typed function 
signatures, error handling — even though the scope is small.

## Context on what's being judged
This is not "just generate an email." The evaluator is equally or more important 
than the generator — treat scoring quality as the core engineering problem, not 
an afterthought. Everything must actually run, not just look good.

## Tech stack (fixed — do not substitute or add other providers/frameworks)
- Language: Python 3.11+
- Primary LLM: Google Gemini API (model: gemini-2.5-flash) via `google-generativeai`
- Fallback LLM: Groq API (model: llama-3.3-70b-versatile) via `groq` SDK
- Validation: pydantic for all structured LLM outputs
- Config: python-dotenv, keys read from .env (never hardcoded)
- No LangChain, no vector DB, no local ML models, no database. Flat JSON files 
  on disk are the only persistence layer.

## Repo structure to create
/data
  - generate_dataset.py
  - dataset.json
/llm
  - client.py              # single shared LLM-call wrapper, used by generator AND evaluator
/generator
  - generate_reply.py
  - prompts.py
/evaluator
  - evaluate.py
  - rubric.py
/results
  - per_response_scores.json
  - aggregate_report.json
run.py
README.md
.env.example
requirements.txt

## 1. Shared LLM client (/llm/client.py) — BUILD THIS FIRST, everything else depends on it
Single function used by both the generator and evaluator, so provider logic lives 
in exactly one place:

def call_llm(prompt: str, response_schema: type[BaseModel] | None = None, 
              max_retries: int = 2) -> str:
    """
    Calls Gemini first. On exception (rate limit, timeout, 5xx, any failure), 
    retries with exponential backoff (base 1.5s) up to max_retries on Gemini.
    If Gemini still fails after retries, falls back once to Groq 
    (llama-3.3-70b-versatile) before raising.
    Logs which provider actually served each call (print or logging module) 
    so failures/fallbacks are visible in stdout during the run, not silent.
    If response_schema is provided, instruct the model via prompt to return 
    strict JSON matching that schema, and validate/parse the response with 
    pydantic before returning. On JSON parse failure, retry once with a 
    stricter "return ONLY valid JSON, no markdown fences" instruction appended.
    """

Requirements:
- Read GEMINI_API_KEY and GROQ_API_KEY from environment (via dotenv)
- Never let one failed call raise uncaught — the caller decides whether to skip 
  or halt, this function's job is retry + fallback + clear error on total failure
- No global mutable state, no singleton client hacks — clean, testable function

## 2. Dataset (/data)
Write a script that generates 15-25 synthetic but realistic customer support email 
scenarios via call_llm(), saved as structured JSON. Vary:
- category (billing, bug report, refund request, feature question, cancellation, complaint)
- customer sentiment (angry, neutral, confused, happy but requesting something)
- urgency (low/medium/high)
- prior_thread_summary for some entries (mimics Hiver's shared-inbox context — not 
  every reply should be to a cold single email)

Schema (define as a pydantic model, reuse for validation):
{
  "id": string,
  "subject": string,
  "customer_email_body": string,
  "category": string,
  "sentiment": string,
  "urgency": string,
  "prior_thread_summary": string | null,
  "ideal_reply_traits": [string]
}
Document in README how the dataset was built and why these fields were chosen.

## 3. Generator (/generator)
Given a dataset entry, generate a structured reply via call_llm() with a pydantic 
response_schema:
{
  "id": string,
  "reply_subject": string,
  "reply_body": string,
  "tone_used": string,
  "suggested_next_action": string,
  "confidence": float (0-1)
}
Requirements:
- Prompt passes ALL context fields (category, sentiment, urgency, prior thread) — 
  not just the raw customer email
- Tone selection driven by sentiment (empathetic for angry/confused, concise for 
  neutral/simple) — make this an explicit prompt instruction, not implicit
- Explicit instruction: do NOT invent policy details, refund amounts, or promises 
  not grounded in the input context
- Process the full dataset with bounded concurrency (asyncio or ThreadPoolExecutor, 
  4-6 workers) — do not loop sequentially, and do not exceed reasonable RPM for 
  either provider
- Wrap each item's generation in try/except so one failure doesn't kill the batch — 
  log failed ids separately and continue

## 4. Evaluator (/evaluator) — THIS IS THE CORE DELIVERABLE, BUILD IT CAREFULLY
Score every generated reply on these independent dimensions, each 0-10:
- relevance — does the reply address what the customer actually asked?
- correctness_faithfulness — no hallucinated facts/policy/promises not grounded 
  in the input context
- tone_appropriateness — does tone match the customer's sentiment?
- completeness — concrete next step / resolution path, not just acknowledgment
- conciseness — penalize unnecessary length/fluff
- safety_compliance — no overpromising, no unsafe claims, no fabricated PII

HYBRID scoring, reported separately per dimension (not merged into one opaque call):
(a) Deterministic/programmatic checks (no LLM call) where possible:
    - conciseness: length bounds against a reasonable word-count range
    - tone_appropriateness: keyword/lexicon-based sentiment match check against 
      input sentiment (simple positive/negative/apologetic keyword sets — no 
      downloaded ML model)
    - completeness: presence-check for action-oriented phrases/next-step markers
(b) LLM-as-judge via call_llm() with response_schema for the more subjective 
    dimensions (relevance, correctness_faithfulness, safety_compliance). Use a 
    STRUCTURED rubric prompt (see rubric.py) that forces the judge to output 
    JSON with a numeric score + one-sentence justification PER dimension in a 
    single call — do not make 6 separate LLM calls per response, batch the judge 
    prompt to cover all LLM-judged dimensions at once to conserve API calls.

Combine into a weighted composite score per response. Weights must be justified 
in README (e.g., correctness_faithfulness + safety_compliance weighted highest — 
factual errors or overpromising in customer support carry real business/legal 
risk; conciseness weighted lowest — soft preference only).

Per-response output schema:
{
  "id": string,
  "scores": {
    "relevance": {"value": float, "method": "llm_judge", "justification": string},
    "correctness_faithfulness": {"value": float, "method": "llm_judge", "justification": string},
    "tone_appropriateness": {"value": float, "method": "deterministic", "detail": string},
    "completeness": {"value": float, "method": "deterministic", "detail": string},
    "conciseness": {"value": float, "method": "deterministic", "detail": string},
    "safety_compliance": {"value": float, "method": "llm_judge", "justification": string}
  },
  "composite_score": float,
  "flags": [string]
}

Also output aggregate_report.json: mean/median per dimension across the dataset, 
worst-performing category, any systemic issues (e.g. "tone mismatch concentrated 
in angry-sentiment entries").

## 5. run.py
Single command, full pipeline: generate dataset (if dataset.json absent) -> 
generate replies for all entries (concurrent, fault-tolerant) -> evaluate all 
replies (concurrent, fault-tolerant) -> write /results -> print summary table 
to stdout (per-dimension averages + composite). Must work on a clean clone with 
only .env configured and `pip install -r requirements.txt`.

## 6. README.md, in this order
1. Approach — why generator and evaluator are separate systems
2. Tech stack — Gemini primary / Groq fallback, and why (reliability under a 
   time-boxed run, not cost — both are free tier)
3. Dataset — how it was built, sample entry
4. Generator — what context is passed and why
5. Why this evaluation metric is right — defend the 6 dimensions and weights, 
   and why hybrid (deterministic + LLM-judge) beats a single "rate 1-10" LLM call
6. Known limitations given the 100-minute constraint — be honest
7. How to run it — exact commands from `git clone` to seeing results, zero 
   assumed local state

## Engineering constraints
- No hardcoded API keys — .env + .env.example checked in (.env in .gitignore)
- No hardcoded local file paths
- Every batch operation (dataset gen, reply gen, evaluation) must tolerate 
  individual item failures without crashing the whole run
- Small, clearly named functions — this will be read by humans
- Provider fallback logic lives ONLY in /llm/client.py — generator and evaluator 
  code must not know or care which provider actually served a call
- After building, simulate a clean clone and run the exact README instructions 
  end-to-end to confirm zero implicit local state

Build in this order, confirming real output after each phase before moving on:
1. /llm/client.py (test with a single hello-world call to both providers)
2. /data (generate and inspect dataset.json)
3. /generator (test on 2-3 entries, inspect output)
4. /evaluator (test on the same 2-3 entries, inspect scores)
5. run.py wiring the full pipeline
6. README

Do not skip step 1 — a broken or untested LLM client wrapper will silently break 
every downstream step.