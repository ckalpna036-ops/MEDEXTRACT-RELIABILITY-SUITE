"""
llm_extractor.py — Pass raw OCR text (.md) through an LLM and extract
clinical entities in the HiLabs JSON schema format.

Supports:
  - Google Gemini 1.5 Flash (free tier, default)
  - OpenAI GPT-4o mini
    - OpenRouter (OpenAI-compatible API)
    - Mistral (OpenAI-compatible API)

Usage (from run_pipeline.py or standalone):
    from src.llm_extractor import extract_entities_from_text
    entities = extract_entities_from_text(raw_text)
"""

import json
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL  = os.getenv("GEMINI_MODEL",  "gemini-1.5-flash")
OPENAI_MODEL  = os.getenv("OPENAI_MODEL",  "gpt-4o-mini")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a clinical NLP expert. Your task is to extract all
clinical entities from a medical document (OCR output from a patient chart or 
medical record) and return them as a structured JSON array.

For EACH entity found, return an object with EXACTLY these fields:
{
  "entity":       "<the extracted text, e.g. aspirin>",
  "entity_type":  "<one of: IMMUNIZATION, MEDICAL_DEVICE, MEDICINE, MENTAL_STATUS, PROBLEM, PROCEDURE, SDOH, SOCIAL_HISTORY, TEST, VITAL_NAME>",
  "assertion":    "<one of: POSITIVE, NEGATIVE, UNCERTAIN>",
  "temporality":  "<one of: CLINICAL_HISTORY, CURRENT, UNCERTAIN, UPCOMING>",
  "subject":      "<one of: PATIENT, FAMILY_MEMBER>",
  "heading":      "<the document section heading where this entity appeared>",
  "text":         "<the surrounding sentence or phrase providing context>",
  "metadata_from_qa": {
    "STRENGTH":         "<value or null>",
    "UNIT":             "<value or null>",
    "DOSE":             "<value or null>",
    "ROUTE":            "<value or null>",
    "FREQUENCY":        "<value or null>",
    "FORM":             "<value or null>",
    "DURATION":         "<value or null>",
    "STATUS":           "<value or null>",
    "exact_date":       "<value or null>",
    "derived_date":     "<value or null>",
    "TEST_VALUE":       "<value or null>",
    "TEST_UNIT":        "<value or null>",
    "VITAL_NAME_UNIT":  "<value or null>",
    "VITAL_NAME_VALUE": "<value or null>"
  }
}

IMPORTANT RULES:
- Return ONLY a valid JSON array — no markdown, no explanation, no extra text.
- Include metadata fields ONLY when the value is present in the text; otherwise set to null.
- Do not invent or hallucinate values not present in the document.
- Use UPPERCASE for all enum values (entity_type, assertion, temporality, subject).
- Extract ALL clinical entities — do not skip any medications, diagnoses, tests, procedures, etc.
"""

USER_PROMPT_TEMPLATE = """Extract all clinical entities from the following medical document:

---
{text}
---

Return only a JSON array of entity objects as described.
"""

# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json_array(raw: str) -> list[dict]:
    """
    Parse the LLM's response robustly — handles markdown code fences,
    trailing commas, and BOM/whitespace.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    # Find JSON array boundaries
    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM response:\n{raw[:500]}")
    json_str = cleaned[start : end + 1]
    # Remove trailing commas before ] or } (common LLM mistake)
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    return json.loads(json_str)


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

def _extract_gemini(text: str, retries: int = 3) -> list[dict]:
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    prompt = USER_PROMPT_TEMPLATE.format(text=text)

    for attempt in range(1, retries + 1):
        try:
            response = model.generate_content(prompt)
            return _extract_json_array(response.text)
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  ⚠️  Gemini attempt {attempt} failed ({e}), retrying in {wait}s…")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _extract_openai(text: str, retries: int = 3) -> list[dict]:
    return _extract_openai_compatible(
        text=text,
        api_key_env="OPENAI_API_KEY",
        model=OPENAI_MODEL,
        provider_label="OpenAI",
    )


def _extract_openrouter(text: str, retries: int = 3) -> list[dict]:
    extra_headers = {}
    site_url = os.getenv("OPENROUTER_SITE_URL", "")
    app_name = os.getenv("OPENROUTER_APP_NAME", "")
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if app_name:
        extra_headers["X-Title"] = app_name

    return _extract_openai_compatible(
        text=text,
        api_key_env="OPENROUTER_API_KEY",
        model=OPENROUTER_MODEL,
        provider_label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        extra_headers=extra_headers or None,
    )


def _extract_mistral(text: str, retries: int = 3) -> list[dict]:
    return _extract_openai_compatible(
        text=text,
        api_key_env="MISTRAL_API_KEY",
        model=MISTRAL_MODEL,
        provider_label="Mistral",
        base_url="https://api.mistral.ai/v1",
    )


def _extract_openai_compatible(
    text: str,
    api_key_env: str,
    model: str,
    provider_label: str,
    retries: int = 3,
    base_url: Optional[str] = None,
    extra_headers: Optional[dict] = None,
) -> list[dict]:
    from openai import OpenAI

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(f"{api_key_env} not set in .env")

    client = OpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers)
    prompt = USER_PROMPT_TEMPLATE.format(text=text)

    for attempt in range(1, retries + 1):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]

            # Try strict JSON mode first where available.
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                )

            raw = (response.choices[0].message.content or "").strip()
            if not raw:
                raise ValueError("Empty response from model")

            # Accept either JSON object wrappers or direct JSON arrays.
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
                for key in ("entities", "data", "results", "items"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                if parsed:
                    first_val = list(parsed.values())[0]
                    if isinstance(first_val, list):
                        return first_val
            except json.JSONDecodeError:
                return _extract_json_array(raw)

            return []
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  ⚠️  {provider_label} attempt {attempt} failed ({e}), retrying in {wait}s…")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities_from_text(text: str) -> list[dict]:
    """
    Run the configured LLM on `text` and return extracted entities.
    Automatically selects backend based on LLM_PROVIDER env var.
    """
    if LLM_PROVIDER == "openai":
        return _extract_openai(text)
    if LLM_PROVIDER == "openrouter":
        return _extract_openrouter(text)
    if LLM_PROVIDER == "mistral":
        return _extract_mistral(text)
    else:
        return _extract_gemini(text)


def extract_entities_from_file(md_path: str) -> list[dict]:
    """Convenience wrapper that loads a .md file then calls extract_entities_from_text."""
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    return extract_entities_from_text(text)
