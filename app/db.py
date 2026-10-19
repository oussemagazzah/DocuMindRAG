"""Database abstraction supporting both SQLite and PostgreSQL."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger("db")

SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'processing',
    created_at TEXT NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    char_start INTEGER NOT NULL DEFAULT 0,
    char_end INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(doc_id, page_number);

CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    doc_ids TEXT,
    answer TEXT,
    model_used TEXT,
    timing_ms REAL,
    groundedness_score REAL,
    groundedness_verdict TEXT,
    citation_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_log_id INTEGER NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('thumbs_up', 'thumbs_down')),
    comment TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (query_log_id) REFERENCES query_logs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_query_log ON feedback(query_log_id);
"""

SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'processing',
    created_at TIMESTAMPTZ NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    char_start INTEGER NOT NULL DEFAULT 0,
    char_end INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(doc_id, page_number);

CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    doc_ids TEXT,
    answer TEXT,
    model_used TEXT,
    timing_ms DOUBLE PRECISION,
    groundedness_score DOUBLE PRECISION,
    groundedness_verdict TEXT,
    citation_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    query_log_id INTEGER NOT NULL REFERENCES query_logs(id) ON DELETE CASCADE,
    rating TEXT NOT NULL CHECK (rating IN ('thumbs_up', 'thumbs_down')),
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_query_log ON feedback(query_log_id);
"""


def _get_db():
    """Get a database connection — returns either aiosqlite or asyncpg."""
    if settings.use_postgres:
        return asyncpg_connection()
    else:
        return aiosqlite_connection()


class aiosqlite_connection:
    """Wrapper around aiosqlite to match asyncpg-like interface."""

    def __init__(self):
        self._db = None
        self._conn_str = settings.database_path

    async def __aenter__(self):
        import aiosqlite
        self._db = await aiosqlite.connect(self._conn_str)
        self._db.row_factory = aiosqlite.Row
        return self

    async def __aexit__(self, *args):
        if self._db:
            await self._db.close()

    async def execute(self, query: str, params: tuple = ()):
        return await self._db.execute(query, params)

    async def fetchall(self):
        rows = await self._db.fetchall()
        return [dict(row) for row in rows]

    async def fetchone(self):
        row = await self._db.fetchone()
        return dict(row) if row else None

    async def commit(self):
        await self._db.commit()

    @property
    def rowcount(self):
        return self._db.total_changes


class asyncpg_connection:
    """Wrapper around asyncpg for PostgreSQL."""

    def __init__(self):
        self._pool = None

    async def __aenter__(self):
        import asyncpg
        self._pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)
        return self

    async def __aexit__(self, *args):
        if self._pool:
            await self._pool.close()

    async def execute(self, query: str, params: tuple = ()):
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *params)

    async def fetchall(self):
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(self._last_query, *self._last_params)
            return [dict(row) for row in rows]

    async def fetchone(self):
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(self._last_query, *self._last_params)
            return dict(row) if row else None

    async def commit(self):
        pass

    @property
    def rowcount(self):
        return 0


async def init_db():
    """Initialize database schema."""
    if settings.use_postgres:
        logger.info("Initializing PostgreSQL database")
        import asyncpg
        conn = await asyncpg.connect(settings.database_url)
        await conn.execute(SCHEMA_POSTGRES)
        await conn.close()
        logger.info("PostgreSQL database initialized")
    else:
        logger.info(f"Initializing SQLite database at {settings.database_path}")
        import aiosqlite
        async with aiosqlite.connect(settings.database_path) as db:
            await db.executescript(SCHEMA_SQLITE)
            await db.commit()
        logger.info("SQLite database initialized")


async def create_document(name: str, page_count: int, chunk_count: int) -> str:
    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    try:
        if settings.use_postgres:
            import asyncpg
            conn = await asyncpg.connect(settings.database_url)
            await conn.execute(
                "INSERT INTO documents (id, name, page_count, chunk_count, status, created_at) VALUES ($1, $2, $3, $4, $5, $6)",
                uuid.UUID(doc_id), name, page_count, chunk_count, "ready", datetime.now(timezone.utc),
            )
            await conn.close()
        else:
            import aiosqlite
            async with aiosqlite.connect(settings.database_path) as db:
                await db.execute(
                    "INSERT INTO documents (id, name, page_count, chunk_count, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, name, page_count, chunk_count, "ready", now),
                )
                await db.commit()
        logger.info(f"Created document: {doc_id} ({name})")
        return doc_id
    except Exception as e:
        logger.error(f"Failed to create document: {e}")
        raise


async def create_document_error(name: str, error_message: str) -> str:
    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    try:
        if settings.use_postgres:
            import asyncpg
            conn = await asyncpg.connect(settings.database_url)
            await conn.execute(
                "INSERT INTO documents (id, name, page_count, chunk_count, status, created_at, error_message) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                uuid.UUID(doc_id), name, 0, 0, "error", datetime.now(timezone.utc), error_message,
            )
            await conn.close()
        else:
            import aiosqlite
            async with aiosqlite.connect(settings.database_path) as db:
                await db.execute(
                    "INSERT INTO documents (id, name, page_count, chunk_count, status, created_at, error_message) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (doc_id, name, 0, 0, "error", now, error_message),
                )
                await db.commit()
        logger.warning(f"Created error document: {doc_id} ({name}): {error_message}")
        return doc_id
    except Exception as e:
        logger.error(f"Failed to create error document: {e}")
        raise


async def create_chunks(doc_id: str, chunks: list[dict]) -> list[str]:
    chunk_ids = []
    try:
        if settings.use_postgres:
            import asyncpg
            conn = await asyncpg.connect(settings.database_url)
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                await conn.execute(
                    "INSERT INTO chunks (id, doc_id, page_number, chunk_index, text, char_start, char_end) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                    uuid.UUID(chunk_id), uuid.UUID(doc_id), chunk["page_number"], chunk["chunk_index"], chunk["text"], chunk["char_start"], chunk["char_end"],
                )
            await conn.close()
        else:
            import aiosqlite
            async with aiosqlite.connect(settings.database_path) as db:
                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    chunk_ids.append(chunk_id)
                    await db.execute(
                        "INSERT INTO chunks (id, doc_id, page_number, chunk_index, text, char_start, char_end) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (chunk_id, doc_id, chunk["page_number"], chunk["chunk_index"], chunk["text"], chunk["char_start"], chunk["char_end"]),
                    )
                await db.commit()
        logger.info(f"Created {len(chunk_ids)} chunks for doc {doc_id}")
        return chunk_ids
    except Exception as e:
        logger.error(f"Failed to create chunks for doc {doc_id}: {e}")
        raise


async def list_documents() -> list[dict]:
    try:
        if settings.use_postgres:
            import asyncpg
            conn = await asyncpg.connect(settings.database_url)
            rows = await conn.fetch("SELECT * FROM documents ORDER BY created_at DESC")
            await conn.close()
            return [dict(row) for row in rows]
        else:
            import aiosqlite
            async with aiosqlite.connect(settings.database_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("SELECT * FROM documents ORDER BY created_at DESC")
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise


async def get_document(doc_id: str) -> Optional[dict]:
    try:
        if settings.use_postgres:
            import asyncpg
            conn = await asyncpg.connect(settings.database_url)
            row = await conn.fetchrow("SELECT * FROM documents WHERE id = $1", uuid.UUID(doc_id))
            await conn.close()
            return dict(row) if row else None
        else:
            import aiosqlite
            async with aiosqlite.connect(settings.database_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
                row = await cursor.fetchone()
                return dict(row) if row else None
    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {e}")
        raise


async def delete_document(doc_id: str) -> bool:
    try:
        if settings.use_postgres:
            import asyncpg
            conn = await asyncpg.connect(settings.database_url)
            await conn.execute("DELETE FROM chunks WHERE doc_id = $1", uuid.UUID(doc_id))
            result = await conn.execute("DELETE FROM documents WHERE id = $1", uuid.UUID(doc_id))
            await conn.close()
            deleted = "DELETE 1" in result
        else:
            import aiosqlite
            async with aiosqlite.connect(settings.database_path) as db:
                await db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                cursor = await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                await db.commit()
                deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted document: {doc_id}")
        else:
            logger.warning(f"Document not found for deletion: {doc_id}")
        return deleted
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise


async def get_chunks_by_doc_id(doc_id: str) -> list[dict]:
    try:
        if settings.use_postgres:
            import asyncpg
            conn = await asyncpg.connect(settings.database_url)
            rows = await conn.fetch(
                "SELECT * FROM chunks WHERE doc_id = $1 ORDER BY page_number, chunk_index", uuid.UUID(doc_id)
            )
            await conn.close()
            return [dict(row) for row in rows]
        else:
            import aiosqlite
            async with aiosqlite.connect(settings.database_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM chunks WHERE doc_id = ? ORDER BY page_number, chunk_index", (doc_id,)
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get chunks for doc {doc_id}: {e}")
        raise


async def log_query(
    query: str,
    doc_ids: list[str] | None,
    answer: str,
    model_used: str,
    timing_ms: float,
    groundedness_score: float,
    groundedness_verdict: str,
    citation_count: int,
) -> int:
    import json as _json
    now = datetime.now(timezone.utc).isoformat()
    doc_ids_str = _json.dumps(doc_ids) if doc_ids else None
    try:
        if settings.use_postgres:
            import asyncpg
            conn = await asyncpg.connect(settings.database_url)
            row = await conn.fetchrow(
                """INSERT INTO query_logs (query, doc_ids, answer, model_used, timing_ms,
                   groundedness_score, groundedness_verdict, citation_count, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id""",
                query, doc_ids_str, answer, model_used, timing_ms,
                groundedness_score, groundedness_verdict, citation_count, datetime.now(timezone.utc),
            )
            await conn.close()
            return row["id"]
        else:
            import aiosqlite
            async with aiosqlite.connect(settings.database_path) as db:
                cursor = await db.execute(
                    """INSERT INTO query_logs (query, doc_ids, answer, model_used, timing_ms,
                       groundedness_score, groundedness_verdict, citation_count, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (query, doc_ids_str, answer, model_used, timing_ms,
                     groundedness_score, groundedness_verdict, citation_count, now),
                )
                await db.commit()
                return cursor.lastrowid
    except Exception as e:
        logger.error(f"Failed to log query: {e}")
        return -1


async def save_feedback(query_log_id: int, rating: str, comment: str | None = None) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    try:
        if settings.use_postgres:
            import asyncpg
            conn = await asyncpg.connect(settings.database_url)
            await conn.execute(
                "INSERT INTO feedback (query_log_id, rating, comment, created_at) VALUES ($1, $2, $3, $4)",
                query_log_id, rating, comment, datetime.now(timezone.utc),
            )
            await conn.close()
        else:
            import aiosqlite
            async with aiosqlite.connect(settings.database_path) as db:
                await db.execute(
                    "INSERT INTO feedback (query_log_id, rating, comment, created_at) VALUES (?, ?, ?, ?)",
                    (query_log_id, rating, comment, now),
                )
                await db.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return False


async def get_analytics_stats() -> dict:
    try:
        if settings.use_postgres:
            import asyncpg
            conn = await asyncpg.connect(settings.database_url)
            total = (await conn.fetchrow("SELECT COUNT(*) as c FROM query_logs"))["c"]
            avg_latency = (await conn.fetchrow("SELECT COALESCE(AVG(timing_ms), 0) as v FROM query_logs"))["v"]
            avg_groundedness = (await conn.fetchrow("SELECT COALESCE(AVG(groundedness_score), 0) as v FROM query_logs"))["v"]
            thumbs_up = (await conn.fetchrow(
                "SELECT COUNT(*) as c FROM feedback WHERE rating = 'thumbs_up'"
            ))["c"]
            thumbs_down = (await conn.fetchrow(
                "SELECT COUNT(*) as c FROM feedback WHERE rating = 'thumbs_down'"
            ))["c"]
            recent = await conn.fetch(
                "SELECT id, query, timing_ms, groundedness_verdict, model_used, created_at FROM query_logs ORDER BY created_at DESC LIMIT 50"
            )
            verdicts = await conn.fetch(
                "SELECT groundedness_verdict, COUNT(*) as c FROM query_logs GROUP BY groundedness_verdict"
            )
            models = await conn.fetch(
                "SELECT model_used, COUNT(*) as c FROM query_logs GROUP BY model_used"
            )
            await conn.close()
            return {
                "total_queries": total,
                "avg_latency_ms": round(float(avg_latency), 2),
                "avg_groundedness": round(float(avg_groundedness), 3),
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "feedback_rate": round((thumbs_up + thumbs_down) / max(total, 1) * 100, 1),
                "recent_queries": [dict(r) for r in recent],
                "verdicts": {r["groundedness_verdict"]: r["c"] for r in verdicts},
                "models_used": {r["model_used"]: r["c"] for r in models},
            }
        else:
            import aiosqlite
            async with aiosqlite.connect(settings.database_path) as db:
                db.row_factory = aiosqlite.Row
                total = (await (await db.execute("SELECT COUNT(*) as c FROM query_logs")).fetchone())["c"]
                avg_latency = (await (await db.execute("SELECT COALESCE(AVG(timing_ms), 0) as v FROM query_logs")).fetchone())["v"]
                avg_groundedness = (await (await db.execute("SELECT COALESCE(AVG(groundedness_score), 0) as v FROM query_logs")).fetchone())["v"]
                thumbs_up = (await (await db.execute("SELECT COUNT(*) as c FROM feedback WHERE rating = 'thumbs_up'")).fetchone())["c"]
                thumbs_down = (await (await db.execute("SELECT COUNT(*) as c FROM feedback WHERE rating = 'thumbs_down'")).fetchone())["c"]
                recent = [dict(r) for r in await (await db.execute(
                    "SELECT id, query, timing_ms, groundedness_verdict, model_used, created_at FROM query_logs ORDER BY created_at DESC LIMIT 50"
                )).fetchall()]
                verdicts = {r["groundedness_verdict"]: r["c"] for r in [dict(r) for r in await (await db.execute(
                    "SELECT groundedness_verdict, COUNT(*) as c FROM query_logs GROUP BY groundedness_verdict"
                )).fetchall()]}
                models_used = {r["model_used"]: r["c"] for r in [dict(r) for r in await (await db.execute(
                    "SELECT model_used, COUNT(*) as c FROM query_logs GROUP BY model_used"
                )).fetchall()]}
                return {
                    "total_queries": total,
                    "avg_latency_ms": round(float(avg_latency or 0), 2),
                    "avg_groundedness": round(float(avg_groundedness or 0), 3),
                    "thumbs_up": thumbs_up,
                    "thumbs_down": thumbs_down,
                    "feedback_rate": round((thumbs_up + thumbs_down) / max(total, 1) * 100, 1),
                    "recent_queries": recent,
                    "verdicts": verdicts,
                    "models_used": models_used,
                }
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        return {
            "total_queries": 0, "avg_latency_ms": 0, "avg_groundedness": 0,
            "thumbs_up": 0, "thumbs_down": 0, "feedback_rate": 0,
            "recent_queries": [], "verdicts": {}, "models_used": {},
        }
