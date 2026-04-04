# core/ingestion/summarizer.py
import json
import re
from anthropic import AsyncAnthropic
from core.config import settings

_client: AsyncAnthropic | None = None

def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client

# IPC → BNS section mapping (extend this as needed)
IPC_TO_BNS = {
    "302": "BNS 103", "304": "BNS 105", "307": "BNS 109",
    "376": "BNS 63",  "420": "BNS 318", "379": "BNS 303",
    "498A": "BNS 85", "406": "BNS 316",
}

SUMMARIZE_PROMPT = """You are an Indian legal document analyst.
Analyse the following document and return ONLY a valid JSON object with these exact keys:
{
  "summary": "2-3 sentence extractive summary using sentences from the document",
  "entities": {"persons": [...], "dates": [...], "locations": [...], "amounts": [...]},
  "sections_mentioned": [{"raw": "Section 302 IPC", "bns_equivalent": "BNS 103"}],
  "doc_type": "FIR|contract|affidavit|judgment|chargesheet|notice|other",
  "keywords": ["max 10 most legally relevant keywords"]
}
Document:
"""


async def call_llm(text: str) -> dict:
    response = await _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": SUMMARIZE_PROMPT + text[:8000]}],
        temperature=0,
    )
    raw = response.content[0].text
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {raw[:200]}") from exc


async def summarize_document(text: str) -> dict:
    result = await call_llm(text)
    # Enrich sections_mentioned with BNS mapping
    for sec in result.get("sections_mentioned", []):
        raw = sec.get("raw", "")
        for ipc_num, bns_equiv in IPC_TO_BNS.items():
            if re.search(rf'\b{re.escape(ipc_num)}\b', raw) and not sec.get("bns_equivalent"):
                sec["bns_equivalent"] = bns_equiv
    return result
