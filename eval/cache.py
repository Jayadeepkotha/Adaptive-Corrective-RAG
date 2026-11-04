import sqlite3
import json
import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Any, Dict, List
from loguru import logger

class EvalCache:
    """
    Persistent SQLite disk cache for LLM completions and retrieval queries.
    Guarantees idempotency and free reruns during benchmark evaluation.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "eval_cache.db"
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_cache (
                    cache_key TEXT PRIMARY KEY,
                    category TEXT,
                    payload TEXT,
                    created_at REAL
                )
            """)
            conn.commit()

    @staticmethod
    def _hash(obj: Any) -> str:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def get_llm(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 0) -> Optional[Dict[str, Any]]:
        key = f"llm:{model}:{max_tokens}:{self._hash(messages)}"
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT payload FROM kv_cache WHERE cache_key = ?", (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def set_llm(self, model: str, messages: List[Dict[str, str]], text: str, usage: Dict[str, int], max_tokens: int = 0):
        key = f"llm:{model}:{max_tokens}:{self._hash(messages)}"
        data = {
            "text": text,
            "usage": usage,
            "timestamp": time.time()
        }
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv_cache (cache_key, category, payload, created_at) VALUES (?, ?, ?, ?)",
                (key, "llm", json.dumps(data), time.time())
            )
            conn.commit()

    def get_retrieval(self, corpus_id: str, query: str, top_k: int, enable_rerank: bool) -> Optional[List[Dict[str, Any]]]:
        key = f"retrieval:{corpus_id}:{self._hash(query)}:{top_k}:{enable_rerank}"
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT payload FROM kv_cache WHERE cache_key = ?", (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def set_retrieval(self, corpus_id: str, query: str, top_k: int, enable_rerank: bool, results: List[Dict[str, Any]]):
        key = f"retrieval:{corpus_id}:{self._hash(query)}:{top_k}:{enable_rerank}"
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv_cache (cache_key, category, payload, created_at) VALUES (?, ?, ?, ?)",
                (key, "retrieval", json.dumps(results), time.time())
            )
            conn.commit()

    def count(self) -> int:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM kv_cache")
            return cur.fetchone()[0]

    def clear(self):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM kv_cache")
            conn.commit()
