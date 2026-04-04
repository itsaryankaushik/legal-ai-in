# core/routing/domain_router.py
import json
from anthropic import AsyncAnthropic
from core.config import settings
from typing import Optional

_client: AsyncAnthropic | None = None

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


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


async def call_classifier(text: str) -> list[dict]:
    response = await _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": ROUTER_PROMPT + text[:2000]}],
        temperature=0,
    )
    raw_text = response.content[0].text
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Classifier returned invalid JSON: {raw_text[:200]}") from exc
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
