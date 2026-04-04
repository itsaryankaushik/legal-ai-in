# core/validation/citation_validator.py
import re


# Patterns for Indian legal citations
CITATION_PATTERNS = [
    r"BNS\s+(?:Section\s+)?\d+[A-Z]?",      # BNS 103, BNS Section 103
    r"BNSS\s+(?:Section\s+)?\d+[A-Z]?",     # BNSS 187
    r"BSA\s+(?:Section\s+)?\d+[A-Z]?",      # BSA 65
    r"IPC\s+(?:Section\s+)?\d+[A-Z]?",      # IPC 302 (legacy)
    r"Section\s+\d+[A-Z]?\s+of\s+the\s+\w+",
    r"Article\s+\d+[A-Z]?",                  # Article 21
]
_PATTERN = re.compile("|".join(CITATION_PATTERNS), re.IGNORECASE)


def extract_citations(text: str) -> list[str]:
    """Extract all legal citations from text."""
    return list(set(_PATTERN.findall(text)))


def _normalise(citation: str) -> str:
    """Normalise citation for comparison: lowercase, remove extra spaces."""
    return re.sub(r"\s+", " ", citation.lower().strip())


def validate_citations(response: str, context: str) -> str:
    """
    For every citation in response, check if it appears in context.
    Citations not found in context are flagged with [UNVERIFIED].
    Returns the response string with [UNVERIFIED] tags appended to bad citations.
    """
    citations = extract_citations(response)
    if not citations:
        return response

    context_normalised = _normalise(context)
    result = response

    for citation in citations:
        if _normalise(citation) not in context_normalised:
            result = result.replace(citation, f"{citation} [UNVERIFIED]")

    return result
