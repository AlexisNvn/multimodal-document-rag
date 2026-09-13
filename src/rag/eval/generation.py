from collections import Counter

from rag.embeddings.text import tokens
from rag.generation.citations import cited_sources


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = Counter(tokens(prediction)), Counter(tokens(reference))
    overlap = sum((predicted & expected).values())
    total = sum(predicted.values()) + sum(expected.values())
    return 2 * overlap / total if total else 1.0


def generation_metrics(answer, references: list[str]) -> dict:
    try:
        used = cited_sources(answer.answer, answer.citations)
        valid = bool(used) or answer.abstained
    except ValueError:
        valid = False
    return {
        "token_f1": max((token_f1(answer.answer, ref) for ref in references), default=0.0),
        "citation_labels_valid": valid,
        "abstained": answer.abstained,
    }
