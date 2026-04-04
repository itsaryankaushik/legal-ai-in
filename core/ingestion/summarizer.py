# core/ingestion/summarizer.py
import json
from openai import AsyncOpenAI
from core.config import settings

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

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
    response = await _client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": SUMMARIZE_PROMPT + text[:8000]}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


async def summarize_document(text: str) -> dict:
    result = await call_llm(text)
    # Enrich sections_mentioned with BNS mapping
    for sec in result.get("sections_mentioned", []):
        raw = sec.get("raw", "")
        for ipc_num, bns_equiv in IPC_TO_BNS.items():
            if ipc_num in raw and not sec.get("bns_equivalent"):
                sec["bns_equivalent"] = bns_equiv
    return result
