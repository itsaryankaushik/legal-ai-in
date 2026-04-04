# core/routing/domain_router.py
import json
import re
from openai import AsyncOpenAI
from core.config import settings
from typing import Optional

_client: AsyncOpenAI | None = None

DOMAINS = [
    "criminal", "constitutional", "civil", "corporate",
    "family", "property", "labour", "tax", "ip", "banking", "cyber"
]

ROUTER_PROMPT = f"""You are an Indian legal domain classifier.
Given a legal document summary or query, classify it into one or more of these domains:
{", ".join(DOMAINS)}

Return ONLY a valid JSON array like:
[{{"domain": "criminal", "confidence": 0.92}}, {{"domain": "constitutional", "confidence": 0.45}}]

Sort by confidence descending. Include only domains with confidence > 0.10.

Text to classify:
"""


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


async def call_classifier(text: str) -> list[dict]:
    response = await _get_client().chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": ROUTER_PROMPT + text[:2000]}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        raw = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Classifier returned invalid JSON: {response.choices[0].message.content[:200]}") from exc
    if isinstance(raw, list):
        return raw
    return raw.get("domains", raw.get("classifications", []))


async def classify_domains(
    text: str,
    redis=None,
) -> list[dict]:
    """Returns sorted list of {domain, confidence} dicts."""
    if redis:
        cached = await redis.get_router_result(text)
        if cached:
            return cached

    result = await call_classifier(text)

    if redis:
        await redis.set_router_result(text, result)

    return result
