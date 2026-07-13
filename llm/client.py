"""
Shared LLM client -- single call_llm() used by generator AND evaluator.

Provider logic lives ONLY here. No other module knows or cares which
provider actually served a call.

Flow: Gemini (primary) -> retry with exp backoff -> Groq (fallback, with retry) -> raise
"""

import json
import os
import re
import time
import threading
from typing import Type

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Thread-safe printing to prevent interleaved output from concurrent workers
_print_lock = threading.Lock()


def _log(msg: str) -> None:
    """Thread-safe log to stdout."""
    with _print_lock:
        print(msg)


def _build_schema_instruction(schema_cls: Type[BaseModel]) -> str:
    """Build a JSON schema instruction string from a pydantic model."""
    schema = schema_cls.model_json_schema()
    return (
        "\n\nYou MUST respond with ONLY valid JSON matching this schema, "
        "no markdown fences, no extra text:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def _extract_json(text: str) -> str:
    """Extract JSON from a response that might have markdown fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _call_gemini(prompt: str) -> tuple[str, float]:
    """Call Gemini API via google.genai, return (response_text, latency_seconds)."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    start = time.time()
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )
    latency = time.time() - start

    return response.text, latency


def _call_groq(prompt: str) -> tuple[str, float]:
    """Call Groq API, return (response_text, latency_seconds)."""
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    start = time.time()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
    )
    latency = time.time() - start

    return response.choices[0].message.content, latency


def call_llm(
    prompt: str,
    response_schema: Type[BaseModel] | None = None,
    max_retries: int = 2,
) -> str:
    """
    Call LLM with retry + fallback logic.

    1. Try Gemini (primary) up to max_retries+1 times with exp backoff
    2. On total Gemini failure, fall back to Groq with 1 retry on 429
    3. If response_schema provided: validate with pydantic, retry once
       on parse failure with stricter instruction
    4. Logs provider, latency, success/failure to stdout (thread-safe)

    Returns raw text (or validated JSON string if schema provided).
    Raises RuntimeError on total failure.
    """
    schema_suffix = ""
    if response_schema:
        schema_suffix = _build_schema_instruction(response_schema)

    full_prompt = prompt + schema_suffix

    # --- Try Gemini with retries ---
    gemini_errors: list[str] = []
    for attempt in range(max_retries + 1):
        try:
            text, latency = _call_gemini(full_prompt)
            _log(f"  [LLM] [OK] Gemini served call ({latency:.1f}s)")

            if response_schema:
                text = _validate_schema(
                    text, response_schema, full_prompt, provider="gemini"
                )
            return text

        except Exception as e:
            error_str = str(e)
            gemini_errors.append(f"attempt {attempt + 1}: {e}")
            _log(
                f"  [LLM] [FAIL] Gemini attempt {attempt + 1}/{max_retries + 1} "
                f"failed: {type(e).__name__} - {error_str[:120]}"
            )
            if attempt < max_retries:
                # If 429 or 503 or resource exhausted, sleep a bit longer
                is_rate_limit = any(x in error_str.upper() for x in ["429", "RESOURCE_EXHAUSTED", "LIMIT", "UNAVAILABLE", "503"])
                if is_rate_limit:
                    # Try to parse retry delay
                    retry_delay = 5.0
                    try:
                        import re as _re
                        delay_match = _re.search(r"retry in ([\d.]+)s", error_str.lower())
                        if delay_match:
                            retry_delay = float(delay_match.group(1)) + 0.5
                    except Exception:
                        pass
                    _log(f"  [LLM] [WARN] Gemini rate-limited/unavailable, waiting {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                else:
                    backoff = 1.5 ** (attempt + 1)
                    _log(f"  [LLM]   Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)

    # --- Fallback to Groq (with 1 retry on rate limit) ---
    _log("  [LLM] >> Falling back to Groq...")
    groq_attempts = 2  # Try Groq up to 2 times
    for groq_attempt in range(groq_attempts):
        try:
            text, latency = _call_groq(full_prompt)
            _log(f"  [LLM] [OK] Groq served call ({latency:.1f}s)")

            if response_schema:
                text = _validate_schema(
                    text, response_schema, full_prompt, provider="groq"
                )
            return text

        except Exception as groq_error:
            error_str = str(groq_error)
            # If rate limited and we have retries left, wait and retry
            if "429" in error_str and groq_attempt < groq_attempts - 1:
                # Parse retry delay from error or use default
                retry_delay = 5.0
                try:
                    import re as _re
                    delay_match = _re.search(r"try again in ([\d.]+)s", error_str)
                    if delay_match:
                        retry_delay = float(delay_match.group(1)) + 0.5
                except Exception:
                    pass
                _log(f"  [LLM] [WARN] Groq rate-limited, waiting {retry_delay:.1f}s...")
                time.sleep(retry_delay)
                continue

            _log(f"  [LLM] [FAIL] Groq fallback failed: {type(groq_error).__name__}")
            all_errors = "; ".join(gemini_errors) + f"; groq: {groq_error}"
            raise RuntimeError(
                f"All LLM providers failed. Errors: {all_errors}"
            ) from groq_error

    # Should not reach here, but safety net
    raise RuntimeError("All LLM providers exhausted.")


def _validate_schema(
    text: str,
    schema_cls: Type[BaseModel],
    original_prompt: str,
    provider: str,
) -> str:
    """
    Validate LLM response against pydantic schema.
    On parse failure, retry once with stricter instruction.
    Returns the validated JSON string.
    """
    cleaned = _extract_json(text)

    try:
        parsed = json.loads(cleaned)
        schema_cls.model_validate(parsed)
        return cleaned
    except Exception as first_error:
        _log(
            f"  [LLM] [WARN] Schema validation failed on {provider}, "
            f"retrying with strict instruction: {first_error}"
        )

    # Retry with stricter instruction
    strict_suffix = (
        "\n\nCRITICAL: Return ONLY valid JSON. No markdown fences, "
        "no explanation, no extra text. Just the raw JSON object."
    )
    strict_prompt = original_prompt + strict_suffix

    try:
        if provider == "gemini":
            text2, latency = _call_gemini(strict_prompt)
        else:
            text2, latency = _call_groq(strict_prompt)

        _log(f"  [LLM] [OK] {provider} strict retry ({latency:.1f}s)")
        cleaned2 = _extract_json(text2)
        parsed2 = json.loads(cleaned2)
        schema_cls.model_validate(parsed2)
        return cleaned2

    except Exception as retry_error:
        raise RuntimeError(
            f"Schema validation failed after strict retry on {provider}: "
            f"{retry_error}. Original error: {first_error}"
        ) from retry_error
