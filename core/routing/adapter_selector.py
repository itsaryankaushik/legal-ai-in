# core/routing/adapter_selector.py
from core.config import settings


DOMAIN_TO_ADAPTER = {
    "criminal": "criminal",
    "constitutional": "constitutional",
    "civil": "civil",
    "corporate": "corporate",
    "family": "family",
    "property": "property",
    "labour": "labour",
    "tax": "tax",
    "ip": "ip",
    "banking": "banking",
    "cyber": "cyber",
}

# All domain adapters available for Phase 1 (zero-shot routing)
# Actual adapter file loading will fail gracefully if a specific adapter file
# is not present on disk — this set controls routing logic only.
AVAILABLE_ADAPTERS = set(DOMAIN_TO_ADAPTER.keys())


def select_adapters(
    domain_scores: list[dict],
    threshold: float = None,
) -> list[str]:
    """
    Return adapter names to activate, sorted by confidence descending.
    Always returns at least one adapter (top-1 regardless of threshold).
    Only activates adapters that exist in AVAILABLE_ADAPTERS.
    """
    if threshold is None:
        threshold = settings.SECONDARY_ADAPTER_CONFIDENCE

    sorted_domains = sorted(domain_scores, key=lambda x: x["confidence"], reverse=True)
    selected = []

    for item in sorted_domains:
        adapter = DOMAIN_TO_ADAPTER.get(item["domain"])
        if adapter and adapter in AVAILABLE_ADAPTERS:
            if item["confidence"] >= threshold or not selected:
                selected.append(adapter)

    return selected if selected else ["criminal"]
