from dataclasses import dataclass

from rag.db.models import Citation


@dataclass
class Context:
    text: str
    citations: list[Citation]
    images: list[tuple[str, str]]


def build_context(hits, max_chars=16000, max_images=4):
    blocks, citations, images = [], [], []
    remaining = max_chars
    seen = set()
    for hit in hits:
        if hit.page.id in seen:
            continue
        seen.add(hit.page.id)
        label = f"S{len(citations) + 1}"
        header = f"[{label}] Document {hit.page.document_id}, page {hit.page.number}\n"
        if remaining <= len(header):
            break
        excerpt = hit.text[: remaining - len(header)]
        use_image = hit.page.image_path and len(images) < max_images
        if not excerpt.strip() and not use_image:
            continue
        block = header + excerpt
        blocks.append(block)
        remaining -= len(block) + 2
        citations.append(
            Citation(label, hit.page.id, hit.page.document_id, hit.page.number, excerpt)
        )
        if use_image:
            images.append((label, hit.page.image_path))
    return Context("\n\n".join(blocks), citations, images)
