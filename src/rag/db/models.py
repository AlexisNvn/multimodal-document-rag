from dataclasses import dataclass, field


@dataclass
class Page:
    id: str
    document_id: str
    number: int  # Human-facing, one-based page number.
    text: str
    image_path: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    page_id: str
    text: str
    parent_text: str
    heading: str = ""


@dataclass
class Hit:
    page: Page
    score: float
    text: str
    channels: list[str] = field(default_factory=list)


@dataclass
class Citation:
    label: str
    page_id: str
    document_id: str
    page_number: int
    quote: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation]
    mode: str
    abstained: bool = False
