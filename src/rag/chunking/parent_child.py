from rag.chunking.structural import sections
from rag.db.models import Chunk, Page


def chunk_page(page: Page, size: int = 180, overlap: int = 30) -> list[Chunk]:
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require size > overlap >= 0")
    result = []
    for heading, parent in sections(page.text):
        words = parent.split()
        for start in range(0, len(words), size - overlap):
            text = " ".join(words[start : start + size])
            result.append(Chunk(f"{page.id}:c{len(result)}", page.id, text, parent, heading))
            if start + size >= len(words):
                break
    return result
