import re


def sections(markdown: str) -> list[tuple[str, str]]:
    """Split on Markdown headings, preserving tables/paragraphs within each section."""
    result = []
    heading, lines = "", []
    for line in markdown.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            if "\n".join(lines).strip():
                result.append((heading, "\n".join(lines).strip()))
            heading, lines = line.lstrip("# ").strip(), [line]
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        result.append((heading, "\n".join(lines).strip()))
    return result
