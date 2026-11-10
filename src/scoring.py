from src import citations


def deterministic_scores(result: dict) -> dict[str, float]:
    draft = result.get("draft", "") or ""
    context = result.get("retrieved_context", "") or ""

    available = citations.extract_available_pages(context)
    cited = citations.extract_cited_pages(draft)
    invalid = [p for p in cited if p not in available]

    if cited:
        citation_validity = 1.0 if not invalid else 0.0
    else:
        citation_validity = 0.0

    return {
        "draft_length": float(len(draft)),
        "has_citations": 1.0 if cited else 0.0,
        "citation_validity": citation_validity,
        "grounded_context_present": 1.0 if len(context) > 200 else 0.0,
        "reflection_passed": 1.0 if result.get("reflection_passed") else 0.0,
    }