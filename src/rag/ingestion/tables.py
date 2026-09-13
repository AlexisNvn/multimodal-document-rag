def table_markdown(table, document) -> str:
    """Keep Docling's table structure rather than flattening cells into prose."""
    return table.export_to_markdown(doc=document).strip()


def rows_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    if not headers or any(len(row) != len(headers) for row in rows):
        raise ValueError("A table requires headers and equally sized rows")

    def line(values):
        return (
            "| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values) + " |"
        )

    return "\n".join([line(headers), line(["---"] * len(headers)), *map(line, rows)])
