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
    Modifier,
)
import re
import uuid
import zlib

from app.config import get_settings
from app.ingestion.chunker import Chunk
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger("embedder")

_model = None
_client = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded")
    return _model


def get_qdrant_client() -> QdrantClient:
    # Cached: an in-memory client loses its data if recreated, and a local
    # folder can only be opened by one client at a time.
    global _client
    if _client is None:
        if settings.qdrant_location == ":memory:":
            _client = QdrantClient(location=":memory:")
        elif settings.qdrant_location:
            _client = QdrantClient(path=settings.qdrant_location)
        else:
            _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=30)
    return _client


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
                    modifier=Modifier.IDF,
                )
            },
        )
        logger.info(f"Collection created: {settings.collection_name}")
        return

    sparse_cfg = client.get_collection(settings.collection_name).config.params.sparse_vectors or {}
    if "sparse" in sparse_cfg and sparse_cfg["sparse"].modifier != Modifier.IDF:
        logger.warning(
            f"Collection {settings.collection_name} was created without the BM25 IDF modifier; "
            "delete it and re-upload documents for correct sparse search."
        )


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


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 or t.isdigit()]


def _token_index(token: str) -> int:
    # crc32 is stable across processes, unlike Python's salted hash().
    return zlib.crc32(token.encode("utf-8"))


def _to_sparse(weights: dict[str, float]) -> SparseVector:
    merged: dict[int, float] = {}
    for token, weight in weights.items():
        idx = _token_index(token)
        merged[idx] = merged.get(idx, 0.0) + weight
    return SparseVector(indices=list(merged), values=list(merged.values()))


def build_sparse_vector(text: str) -> SparseVector:
    """Document-side BM25 vector: saturated, length-normalised term frequency.

    The IDF factor is applied by Qdrant at query time (Modifier.IDF), so the
    stored value for term t is tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len)).
    """
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])
    k1, b = settings.bm25_k1, settings.bm25_b
    norm = k1 * (1 - b + b * len(tokens) / settings.bm25_avg_doc_len)
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return _to_sparse({t: tf * (k1 + 1) / (tf + norm) for t, tf in counts.items()})


def build_sparse_query_vector(query: str) -> SparseVector:
    """Query-side BM25 vector: each distinct query term with weight 1 (Qdrant adds IDF)."""
    return _to_sparse({t: 1.0 for t in set(tokenize(query))})


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
