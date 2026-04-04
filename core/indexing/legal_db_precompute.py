# core/indexing/legal_db_precompute.py
import json
from pathlib import Path
from openai import AsyncOpenAI
from core.config import settings

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


LEGAL_SUMMARY_PROMPT = """You are an Indian legal expert.
Summarise this legal section in 2-3 sentences for a lawyer.
Include: what the section covers, what it prohibits/permits, what punishment/remedy it provides.
Return ONLY the summary.

Section text:
"""


async def generate_section_summary(section_text: str) -> str:
    response = await _get_client().chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": LEGAL_SUMMARY_PROMPT + section_text[:2000]}],
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


async def build_legal_node(section: dict) -> dict:
    """Convert a raw section dict to a PageIndex node with LLM summary."""
    summary = await generate_section_summary(section["text"])
    act_code = section["act"].replace(" ", "_").replace(",", "")[:20]
    node_id = f"{act_code}-{section['section_number']}"
    return {
        "node_id": node_id,
        "title": f"Section {section['section_number']} — {section['title']}",
        "act": section["act"],
        "section_number": section["section_number"],
        "old_equivalent": section.get("old_equivalent", ""),
        "summary": summary,
        "text": section["text"],
        "keywords": section.get("keywords", []),
        "domain": section["domain"],
        "page_range": section.get("page_range", []),
        "sub_nodes": [],
    }


async def precompute_from_file(json_path: str) -> list[dict]:
    """Load sections from JSON, generate summaries, return list of nodes."""
    sections = json.loads(Path(json_path).read_text())
    nodes = []
    for section in sections:
        node = await build_legal_node(section)
        nodes.append(node)
        print(f"  Built node: {node['node_id']}")
    return nodes
