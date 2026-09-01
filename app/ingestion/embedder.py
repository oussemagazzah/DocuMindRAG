"""Embedding generation and Qdrant vector store operations."""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    SparseIndexParams,
)
import uuid

from app.config import get_settings
from app.ingestion.chunker import Chunk
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger("embedder")

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded")
    return _model


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=30)


def ensure_collection(client: QdrantClient):
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if settings.collection_name not in collection_names:
        logger.info(f"Creating Qdrant collection: {settings.collection_name}")
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                )
            },
        )
        logger.info(f"Collection created: {settings.collection_name}")


def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    model = get_embedding_model()
    texts = [c.text for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks")
    embeddings = model.encode(texts, show_progress_bar=False)
    return list(zip(chunks, embeddings.tolist()))


def upsert_chunks(doc_id: str, embedded_chunks: list[tuple[Chunk, list[float]]]) -> int:
    client = get_qdrant_client()
    ensure_collection(client)

    points = []
    for chunk, embedding in embedded_chunks:
        chunk_id = str(uuid.uuid4())
        sparse = build_sparse_vector(chunk.text)

        points.append(
            PointStruct(
                id=chunk_id,
                vector={
                    "dense": embedding,
                    "sparse": sparse,
                },
                payload={
                    "doc_id": doc_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
            )
        )

    client.upsert(collection_name=settings.collection_name, points=points)
    logger.info(f"Upserted {len(points)} vectors for doc {doc_id}")
    return len(points)


def build_sparse_vector(text: str) -> SparseVector:
    tokens = text.lower().split()
    token_counts: dict[str, int] = {}
    for token in tokens:
        clean = token.strip(".,;:!?\"'()-")
        if clean and len(clean) > 1:
            token_counts[clean] = token_counts.get(clean, 0) + 1

    seen_indices: set[int] = set()
    indices = []
    values = []
    for token, count in token_counts.items():
        idx = hash(token) % 100000
        while idx in seen_indices:
            idx = (idx + 1) % 100000
        seen_indices.add(idx)
        indices.append(idx)
        values.append(float(count))

    return SparseVector(indices=indices, values=values)


def delete_document_vectors(doc_id: str):
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_qdrant_client()
    client.delete(
        collection_name=settings.collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        ),
    )
    logger.info(f"Deleted vectors for doc {doc_id}")
