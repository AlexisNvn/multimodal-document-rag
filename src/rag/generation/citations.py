import re


def cited_sources(answer: str, citations: list):
    labels = set(re.findall(r"\[(S\d+)\]", answer))
    allowed = {citation.label for citation in citations}
    if labels - allowed:
        raise ValueError(f"Unknown citation labels: {sorted(labels - allowed)}")
    return [citation for citation in citations if citation.label in labels]
