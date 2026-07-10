import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def split_env_list(name: str) -> List[str]:
    return [item.strip().lower() for item in os.getenv(name, "").split(",") if item.strip()]


def split_env_csv(name: str, default: List[str]) -> List[str]:
    configured = [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]
    return configured or default


def get_groq_api_keys() -> List[str]:
    """Read Groq API keys from .env/environment, preserving priority order."""
    candidates = []
    candidates.extend(split_env_csv("GROQ_API_KEYS", []))
    candidates.extend([
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("GROQ_API_Key", ""),
    ])
    for i in range(1, 6):
        candidates.append(os.getenv(f"GROQ_API_KEY_{i}", ""))

    keys = []
    seen = set()
    for key in candidates:
        key = key.strip()
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def get_database_url() -> str:
    configured = os.getenv("DATABASE_URL", "").strip()
    if configured:
        if configured.startswith("postgresql://"):
            return configured.replace("postgresql://", "postgresql+psycopg://", 1)
        return configured
    return f"sqlite:///{PROJECT_ROOT / 'data' / 'app.db'}"


CONFIG = {
    "provider": "groq",
    "faiss_index_file": str(PROJECT_ROOT / "data" / "bal_faiss.index"),
    "chunks_meta_file": str(PROJECT_ROOT / "data" / "bal_chunks.json"),
    "embedding_model": "intfloat/multilingual-e5-small",
    "retrieval_top_k": 5,
    "retrieval_score_threshold": 0.35,
    "groq_url": "https://api.groq.com/openai/v1/chat/completions",
    "groq_model_chain": split_env_csv("GROQ_MODEL_CHAIN", [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]),
    "groq_api_keys": get_groq_api_keys(),
    "groq_timeout": 120,
    "llm_temperature": 0.1,
    "llm_max_tokens": 1024,
    "llm_top_p": 0.9,
    "max_history_turns": 6,
    "database_url": get_database_url(),
    "secret_key": os.getenv("FLASK_SECRET_KEY", ""),
    "force_https": env_bool("FORCE_HTTPS", False),
    "local_https": env_bool("LOCAL_HTTPS", False),
    "google_client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
    "admin_emails": split_env_list("ADMIN_EMAILS"),
    "limits": {
        "visitor": {"daily": 40, "minute": 5},
        "user": {"daily": 50, "minute": 8},
        "admin": {"daily": 500, "minute": 20},
    },
    "embedding_max_workers": 2,
    "congestion_threshold": 4,
}
