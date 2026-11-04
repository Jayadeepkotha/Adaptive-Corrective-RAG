import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure fast offline initialization for cached Hugging Face models
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # Models & Providers
    LLM_PROVIDER: str = "groq"  # "groq" (fast, active) with Gemini fallback
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434/v1"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    OLLAMA_TIMEOUT: float = 1.0
    LLM_MODEL: str = "openai/gpt-oss-20b"
    FALLBACK_MODEL: str = "openai/gpt-oss-120b"
    GEMINI_MODEL: str = "gemini-3.8-flash"
    GEMINI_FALLBACK_MODEL: str = "gemini-3.8-flash"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Retrieval & Reranker Settings
    TOP_K: int = 4
    CANDIDATE_POOL_K: int = 15
    SIMILARITY_THRESHOLD: float = 0.40
    CROSS_ENCODER_CALIBRATION_OFFSET: float = 8.5
    MAX_PER_DOC: Optional[int] = None

    # CRAG Evaluator Settings
    CRAG_UPPER_THRESHOLD: float = 0.60
    CRAG_LOWER_THRESHOLD: float = 0.25

    # Ablation & Feature Flags
    ENABLE_GRAPH_RAG: bool = True
    ENABLE_RERANK: bool = True
    ENABLE_WEB_SEARCH: bool = True

    # Paths
    DATA_DIR: Path = ROOT_DIR / "data"
    DOCS_DIR: Path = ROOT_DIR / "data" / "docs"

settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
