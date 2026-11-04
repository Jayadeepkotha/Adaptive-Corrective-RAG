import os
import sqlite3
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import faiss
from loguru import logger
from src.config import settings

class VaultStorage:
    """
    Enterprise Persistent Storage Layer:
    1. FAISS IndexFlatIP persistence for dense embeddings.
    2. SQLite database for document metadata, chunk text, and knowledge graph triples.
    Ensures that uploaded documents and knowledge graph connections persist across server restarts.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "vault.db"
        self.faiss_path = self.data_dir / "vault.faiss"
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_name TEXT PRIMARY KEY,
                    doc_type TEXT,
                    chunk_count INTEGER,
                    created_at REAL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_name TEXT,
                    text TEXT,
                    page INTEGER,
                    vector_idx INTEGER,
                    created_at REAL,
                    FOREIGN KEY (doc_name) REFERENCES documents(doc_name) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT,
                    relation TEXT,
                    object TEXT,
                    doc_name TEXT,
                    created_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_name);
                CREATE INDEX IF NOT EXISTS idx_triples_doc ON triples(doc_name);
            """)

    def save_batch(
        self,
        doc_name: str,
        doc_type: str,
        chunks: List[Any],  # List[DocumentChunk]
        embeddings: np.ndarray,
        triples: Optional[List[Tuple[str, str, str]]] = None,
    ):
        """Saves chunks, embeddings, and knowledge graph triples to SQLite and FAISS."""
        if not chunks:
            return

        now = time.time()
        triples = triples or []

        # 1. Update FAISS Index
        dim = embeddings.shape[1]
        faiss_index = self.load_faiss_index(dim=dim)
        start_idx = faiss_index.ntotal

        # Ensure embeddings are normalized float32
        norm_embs = embeddings.astype("float32").copy()
        faiss.normalize_L2(norm_embs)
        faiss_index.add(norm_embs)
        faiss.write_index(faiss_index, str(self.faiss_path))

        # 2. Persist to SQLite
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO documents (doc_name, doc_type, chunk_count, created_at) VALUES (?, ?, ?, ?)",
                (doc_name, doc_type, len(chunks), now)
            )

            # Insert chunks
            chunk_records = []
            for i, c in enumerate(chunks):
                vector_idx = start_idx + i
                chunk_records.append((c.chunk_id, doc_name, c.text, c.page, vector_idx, now))

            cursor.executemany(
                "INSERT OR REPLACE INTO chunks (chunk_id, doc_name, text, page, vector_idx, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                chunk_records
            )

            # Insert triples
            if triples:
                triple_records = []
                for item in triples:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        s, r, o = item[0], item[1], item[2]
                        if s and o:
                            triple_records.append((str(s).strip(), str(r).strip().upper(), str(o).strip(), doc_name, now))
                if triple_records:
                    cursor.executemany(
                        "INSERT INTO triples (subject, relation, object, doc_name, created_at) VALUES (?, ?, ?, ?, ?)",
                        triple_records
                    )
            conn.commit()

        logger.info(f"Persisted '{doc_name}' ({len(chunks)} chunks, {len(triples)} triples) to vault storage.")

    def load_faiss_index(self, dim: int = 384) -> faiss.IndexFlatIP:
        """Loads FAISS index from disk or creates a new empty one."""
        if self.faiss_path.exists():
            try:
                return faiss.read_index(str(self.faiss_path))
            except Exception as e:
                logger.warning(f"Error loading FAISS index: {e}. Reinitializing new index...")
        return faiss.IndexFlatIP(dim)

    def load_all_chunks(self) -> List[Dict[str, Any]]:
        """Loads all chunks ordered by vector index."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id, doc_name, text, page, vector_idx FROM chunks ORDER BY vector_idx ASC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def load_all_triples(self) -> List[Tuple[str, str, str, str]]:
        """Loads all knowledge graph triples (subject, relation, object, doc_name)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT subject, relation, object, doc_name FROM triples")
            rows = cursor.fetchall()
            return [(r["subject"], r["relation"], r["object"], r["doc_name"] or "") for r in rows]

    def list_documents(self) -> List[Dict[str, Any]]:
        """Lists all indexed documents and their chunk counts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT doc_name, doc_type, chunk_count, created_at FROM documents ORDER BY created_at DESC")
            return [dict(r) for r in cursor.fetchall()]

    def clear(self):
        """Clears all persistent data and deletes the FAISS index."""
        with self._get_connection() as conn:
            conn.executescript("""
                DELETE FROM triples;
                DELETE FROM chunks;
                DELETE FROM documents;
            """)
        if self.faiss_path.exists():
            try:
                self.faiss_path.unlink()
            except Exception as e:
                logger.warning(f"Error deleting FAISS file: {e}")
        logger.info("Cleared persistent vault storage.")
