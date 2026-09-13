import json
from threading import RLock

from rag.db.repositories import Repository
from rag.embeddings.text import HashEncoder, SentenceTransformerEncoder
from rag.generation.answer import chat_answer, extractive_answer
from rag.generation.context import build_context
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval import dense, sparse, visual
from rag.retrieval.fusion import reciprocal_rank_fusion


class RAGService:
    def __init__(self, settings):
        self.settings = settings
        self.lock = RLock()
        self.repository = Repository(settings.data_dir / "rag.sqlite3")
        signature = json.dumps(
            {
                "version": 1,
                "text_backend": settings.text_backend,
                "text_model": settings.text_model,
                "visual_backend": settings.visual_backend,
                "visual_model": settings.visual_model,
                "chunk_words": settings.chunk_words,
                "chunk_overlap": settings.chunk_overlap,
                "parser": settings.parser,
            },
            sort_keys=True,
        )
        self.repository.configure(signature)
        self.text_encoder = (
            HashEncoder()
            if settings.text_backend == "hash"
            else SentenceTransformerEncoder(settings.text_model, settings.device)
        )
        self.visual_encoder = None
        if settings.visual_backend == "colqwen2":
            from rag.embeddings.visual import ColQwen2Encoder

            self.visual_encoder = ColQwen2Encoder(settings.visual_model, settings.device)
        self.reranker = None
        if settings.reranker_model:
            from rag.retrieval.reranker import CrossEncoderReranker

            self.reranker = CrossEncoderReranker(settings.reranker_model, settings.device)
        self.pipeline = IngestionPipeline(
            self.repository, self.text_encoder, settings, self.visual_encoder
        )

    def retrieve(self, query, k=5, mode="hybrid", document_id=None):
        if not query.strip() or not 1 <= k <= 100:
            raise ValueError("A nonempty query and k between 1 and 100 are required")
        if mode not in {"dense", "sparse", "visual", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        if mode == "visual" and self.visual_encoder is None:
            raise ValueError("Visual retrieval is disabled; configure RAG_VISUAL_BACKEND")
        with self.lock:
            pages = {
                key: page
                for key, page in self.repository.pages().items()
                if document_id is None or page.document_id == document_id
            }
            chunks = [
                (chunk, vector)
                for chunk, vector in self.repository.chunks()
                if chunk.page_id in pages
            ]
            rankings = []
            candidates = min(100, max(20, k * 4))
            if mode in {"dense", "hybrid"}:
                rankings.append(dense.search(query, chunks, pages, self.text_encoder, candidates))
            if mode in {"sparse", "hybrid"}:
                rankings.append(sparse.search(query, chunks, pages, candidates))
            if mode in {"visual", "hybrid"} and self.visual_encoder:
                vectors = [
                    (key, vector)
                    for key, vector in self.repository.visual_vectors()
                    if key in pages
                ]
                rankings.append(
                    visual.search(query, vectors, pages, self.visual_encoder, candidates)
                )
            hits = reciprocal_rank_fusion(rankings, candidates) if mode == "hybrid" else rankings[0]
            if self.reranker:
                hits = self.reranker.rerank(query, hits)
            return hits[:k]

    def answer(self, query, k=5, mode="hybrid", document_id=None):
        hits = self.retrieve(query, k, mode, document_id)
        context = build_context(hits, self.settings.context_chars)
        answer = (
            chat_answer(query, context, self.settings)
            if self.settings.answer_backend == "chat"
            else extractive_answer(context)
        )
        return answer, hits
