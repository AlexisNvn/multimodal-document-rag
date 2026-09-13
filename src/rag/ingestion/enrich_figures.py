from collections.abc import Callable
from pathlib import Path


def enrich_figures(document, output: Path, captioner: Callable | None = None) -> dict:
    """Export figures and optional descriptions, retaining provenance and generated labels."""
    output.mkdir(parents=True, exist_ok=True)
    by_page = {}
    for index, picture in enumerate(document.pictures):
        image = picture.get_image(document)
        if image is None or not picture.prov:
            continue
        path = output / f"figure-{index}.png"
        image.save(path)
        record = {
            "image_path": str(path.resolve()),
            "caption": picture.caption_text(document),
            "bbox": picture.prov[0].bbox.model_dump(mode="json"),
        }
        if captioner is not None:
            record["generated_description"] = captioner(image)
        by_page.setdefault(picture.prov[0].page_no, []).append(record)
    return by_page
