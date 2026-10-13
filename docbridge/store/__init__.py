import os

from .memory import MemoryStore


def default_store(dim: int = 768):
    """pgvector when DOCBRIDGE_DATABASE_URL is set, otherwise in memory."""
    dsn = os.environ.get("DOCBRIDGE_DATABASE_URL")
    if dsn:
        from .pgvector import PgVectorStore

        return PgVectorStore(dsn, dim=dim)
    return MemoryStore()


__all__ = ["MemoryStore", "default_store"]
