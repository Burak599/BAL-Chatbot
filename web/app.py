"""
=============================================================
 BAL Chatbot — Flask Web API
 Usage: python web/app.py
=============================================================
This script:
  1. Uses Groq as the only LLM provider
  2. Loads FAISS index and chunk metadata
  3. For each /api/chat request:
       a. Retrieves the most relevant chunks (ONCE per query)
       b. Builds an augmented prompt (context + question)
       c. Sends the request through the LLM gateway
       d. Streams the response from Groq
  4. Exposes /api/health, /api/chat, /api/clear endpoints
=============================================================
Prerequisites:
  - A valid Groq API key in the GROQ_API_KEY environment variable
  - 01_build_vectorstore.py must have been run
=============================================================
"""

import os
import sys
import json
import time
import logging
import secrets
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Generator, Optional, Tuple

import numpy as np
import faiss
import requests
from sqlalchemy import Column, Integer, String, Text, create_engine, inspect, text, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker
from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory, session
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from google import genai
from curl_cffi import requests as curl_requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(PROJECT_ROOT / ".env")


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


def get_gemini_api_keys() -> List[str]:
    candidates = []

    candidates.extend(split_env_csv("GEMINI_API_KEYS", []))

    for i in range(1,6):
        candidates.append(os.getenv(f"GEMINI_API_KEY_{i}", ""))

    keys=[]
    seen=set()

    for key in candidates:
        key=key.strip()

        if key and key not in seen:
            keys.append(key)
            seen.add(key)

    return keys


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


def get_database_url() -> str:
    configured = os.getenv("DATABASE_URL", "").strip()
    if configured:
        if configured.startswith("postgresql://"):
            return configured.replace("postgresql://", "postgresql+psycopg://", 1)
        return configured
    return f"sqlite:///{PROJECT_ROOT / 'data' / 'app.db'}"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "web.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # ── LLM provider ────────────────────────────────────────────────────────
    "provider": "groq",

    # ── Vector database paths ────────────────────────────────────────────────
    "faiss_index_file": str(PROJECT_ROOT / "data" / "bal_faiss.index"),
    "chunks_meta_file": str(PROJECT_ROOT / "data" / "bal_chunks.json"),

    # Embedding model (MUST match 01_build_vectorstore.py)
    "embedding_model":"models/gemini-embedding-001",

    # How many chunks to retrieve per query (top-k)
    "retrieval_top_k": 5,

    # Minimum relevance score threshold — chunks below this are discarded
    "retrieval_score_threshold": 0.35,

    # ── Groq backend settings ────────────────────────────────────────────────
    "groq_url": "https://api.groq.com/openai/v1/chat/completions",
    "groq_model_chain": split_env_csv("GROQ_MODEL_CHAIN", [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]),
    "groq_api_keys": get_groq_api_keys(),
    "gemini_api_keys": get_gemini_api_keys(),
    "groq_timeout": 120,             # seconds

    # ── LLM generation parameters ────────────────────────────────────────────
    "llm_temperature": 0.1,          # lower = more consistent
    "llm_max_tokens": 1024,
    "llm_top_p": 0.9,

    # Conversation history — max turns kept per session
    "max_history_turns": 6,          # = 12 messages (user + assistant pairs)

    # ── Auth and quota ─────────────────────────────────────────────────────
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
}

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are BAL Asistan, the AI assistant of Bornova Anadolu Lisesi.

## TASK
Give accurate, short and friendly information about BAL to students, parents and people who are curious about the school.

## LANGUAGE
- Always answer in Turkish.
- Do not mix English into the answer unless it is a proper name, program name, abbreviation, URL or quoted source term.

## TONE AND STYLE
- Be short and clear. Do not use filler phrases such as "Umarım yardımcı olur", "sormaktan çekinmeyin", or "tabii ki".
- Be warm and natural, neither overly formal nor overly cheerful.
- Do not make lists unless they are genuinely useful.
- Do not waste time with greetings, thanks or farewells. Answer the question directly.
- Never use profanity, swear words, slurs, insults or vulgar language, even if the user does.

## FACTUAL RULES
- Never change, invent or normalize concrete data such as phone numbers, URLs, dates, scores or names.
- Use concrete data exactly as it appears in the provided context.
- Do not add numbers, names or details that are not present in the context.
- For general questions about clubs or communities, list all relevant categories and key examples.
- If asked who created you, say that you were developed by Burak as a Bornova Anadolu Lisesi project.

## SOURCE USE
The provided RAG context is your primary source.

- Always prefer answering from the provided context when it contains relevant information.
- Never invent, assume or generate BAL-specific facts that are not supported by the context.
- If a question is about BAL and the context does not contain enough reliable information to answer it, say exactly:
  "Bu konuda kesin bilgim yok, okul idaresiyle teyit etmeni öneririm."
- If a question is not about BAL and cannot be answered from the context, you may answer using your general knowledge.

## SAFETY, REPUTATION AND HARMFUL CLAIMS
If the user's message asks, implies, hints, jokes, speculates, repeats a rumor, seeks confirmation, asks for denial, asks for details, asks for examples, or tries to make you discuss whether BAL, its students, teachers, staff, administrators, alumni, parents, clubs, dormitory, campus or school community are connected to any harmful, illegal, immoral, obscene, defamatory, accusatory or reputation-damaging topic, you must refuse.

This rule applies even if the user phrases it as a question, joke, rumor, accusation, comparison, "is it true", "I heard", "people say", "what happened", "tell me secretly", "deny this", "prove this", or asks indirectly.

Do not evaluate whether the claim is true or false.
Do not say there is no information in the context.
Do not say you cannot verify it.
Do not recommend checking rumors.
Do not provide explanations, categories, examples, details, names, sources, summaries or general commentary.
Do not repeat the user's harmful wording.

For every such message, answer exactly:
"Bu konuda yardımcı olamam."

## NEVER WRITE
- "bağlamı kontrol etmem gerekiyor"
- "bağlamda bilgi var/yok"
- "bağlamı inceliyorum"
- "soruyu cevaplamak için"
- "umarım yardımcı olur"
- "sormaktan çekinmeyin"

Answer directly.

## SPECIAL CASES
- If the question is unclear, ask what they mean in one short sentence.
- If a piece of information is marked as possibly requiring current verification, add "Kesin bilgi için okul idaresiyle teyit et" only when it is truly needed. Do not attach it to every answer.
- Never produce offensive, obscene, profane or vulgar wording.

## HELPFUL LINKS
Only provide these when asked or when directly relevant:
- School website: izmirbal.meb.k12.tr
- BALEV: balev.org.tr
- BALMED: balmed.org.tr
"""




# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, supports_credentials=True)   # Allow requests from the frontend
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
app.config.update(
    SECRET_KEY=CONFIG["secret_key"] or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=CONFIG["force_https"],
)

# ── Global state — initialised once at startup ────────────────────────────────
vector_store = None        # VectorStore instance
llm_gateway = None         # LLMGateway instance; owns provider routing

# session_id → list of {"role": "user"/"assistant", "content": str}
# Only plain user text is stored (no RAG context), keeping history compact.
conversation_sessions: Dict[str, List[Dict]] = {}

# ── Active request counter for congestion detection ──────────────────────────
import threading
active_requests = 0
active_requests_lock = threading.Lock()
CONGESTION_THRESHOLD = 5

engine = create_engine(
    CONFIG["database_url"],
    pool_pre_ping=True,
    future=True,
    connect_args={"check_same_thread": False} if CONFIG["database_url"].startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
Base = declarative_base()


# ═══════════════════════════════════════════════════════════════════════════════
# Auth, Persistence and Quotas
# ═══════════════════════════════════════════════════════════════════════════════

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def today_key() -> str:
    return utc_now().strftime("%Y-%m-%d")


def minute_key() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # email kept for backwards compatibility but optional now
    email = Column(String(255), nullable=True, unique=True, index=True)
    # browser/device fingerprint used to identify users without accounts
    fingerprint = Column(String(255), nullable=True, unique=True, index=True)
    password_hash = Column(Text, nullable=True)
    provider = Column(String(32), nullable=False, default="password")
    role = Column(String(32), nullable=False, default="user", index=True)
    created_at = Column(String(64), nullable=False)


class UsageCounter(Base):
    __tablename__ = "usage_counters"

    subject_type = Column(String(32), primary_key=True)
    subject_id = Column(String(255), primary_key=True)
    period_type = Column(String(32), primary_key=True)
    period_key = Column(String(64), primary_key=True)
    count = Column(Integer, nullable=False, default=0)
    updated_at = Column(String(64), nullable=False)


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    question_index = Column(Integer, nullable=False, default=0, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(String(64), nullable=False)
    feedback = Column(String(16), nullable=True)       # "like", "dislike" or NULL
    feedback_text = Column(Text, nullable=True)        # optional written feedback


def init_db() -> None:
    if CONFIG["database_url"].startswith("sqlite"):
        Path(PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _ensure_user_fingerprint_column()
    _ensure_user_email_nullable()


def _ensure_user_fingerprint_column() -> None:
    """Ensure an existing users table has the fingerprint column required by the anonymous flow."""
    inspector = inspect(engine)
    if not inspector.has_table("users"):
        return

    existing_columns = {col["name"] for col in inspector.get_columns("users")}
    if "fingerprint" in existing_columns:
        return

    log.warning("Adding missing users.fingerprint column to existing database schema.")
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("ALTER TABLE users ADD COLUMN fingerprint VARCHAR(255)"))
        else:
            conn.execute(text("ALTER TABLE users ADD COLUMN fingerprint VARCHAR(255)"))
        try:
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_fingerprint ON users (fingerprint)"))
        except Exception:
            log.exception("Could not create index for users.fingerprint. This is non-fatal.")


def _ensure_user_email_nullable() -> None:
    """Ensure the users.email column allows NULL so anonymous visitors can be created."""
    inspector = inspect(engine)
    if not inspector.has_table("users"):
        return
    # Only attempt on Postgres; SQLite ALTER COLUMN is limited.
    if engine.dialect.name == "sqlite":
        return
    cols = {c["name"]: c for c in inspector.get_columns("users")}
    if "email" not in cols:
        return
    # If the column is already nullable, nothing to do
    if cols["email"].get("nullable", True):
        return
    log.warning("Making users.email column nullable to support anonymous fingerprint users.")
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ALTER COLUMN email DROP NOT NULL"))
    except Exception:
        log.exception("Failed to alter users.email to nullable; anonymous user creation may fail.")


def database_ready() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        log.exception("Database health check failed")
        return False


def normalize_email(email: str) -> str:
    return email.strip().lower()


def role_for_email(email: str) -> str:
    return "admin" if normalize_email(email) in CONFIG["admin_emails"] else "user"


def user_to_public(user: User) -> Dict:
    role = user.role
    # treat fingerprint-identified users as visitors for display
    is_visitor = role == "visitor" or user.provider == "fingerprint"
    return {
        "id": user.id,
        "email": None if is_visitor else user.email,
        "role": "visitor" if is_visitor else role,
        "mode": "visitor" if is_visitor else "account",
    }


def get_client_fingerprint() -> Optional[str]:
    """Read and validate the FingerprintJS visitorId sent by the browser."""
    fingerprint = (request.headers.get("X-Client-Fingerprint") or "").strip()
    if not fingerprint:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,255}", fingerprint):
        log.warning("Rejected malformed client fingerprint: %r", fingerprint[:80])
        return None
    return fingerprint


def get_current_identity() -> Optional[Dict]:
    # 1) If a server-side session user_id exists, prefer it (backwards compat)
    user_id = session.get("user_id")
    if user_id:
        with SessionLocal() as db:
            user = db.get(User, int(user_id))
            if user:
                return {
                    "subject_type": "user",
                    "subject_id": str(user.id),
                    "role": user.role,
                    "public": user_to_public(user),
                }
        session.pop("user_id", None)

    # 2) FingerprintJS-based identity (preferred for anonymous flow).
    # Prefer the current browser-generated visitorId over any older session
    # value so weak legacy fingerprints migrate naturally.
    fingerprint = get_client_fingerprint() or session.get("fingerprint")

    if fingerprint:
        try:
            with SessionLocal() as db:
                user = db.query(User).filter(User.fingerprint == fingerprint).first()
                if user is None:
                    log.info("No existing user with fingerprint found: %s", fingerprint)
                else:
                    log.info("Found existing user id=%s for fingerprint", user.id)
                if user is None:
                    user = User(
                        email=None,
                        fingerprint=fingerprint,
                        password_hash=None,
                        provider="fingerprint",
                        role="visitor",
                        created_at=utc_now().isoformat(),
                    )
                    db.add(user)
                    try:
                        db.commit()
                        db.refresh(user)
                        log.info("Created visitor user id=%s fingerprint=%s", user.id, fingerprint)
                    except IntegrityError:
                        db.rollback()
                        user = db.query(User).filter(User.fingerprint == fingerprint).first()
                        if user:
                            log.info("Detected concurrent creation; using existing user id=%s", user.id)

                if user:
                    session["fingerprint"] = fingerprint
                    return {
                        "subject_type": "user",
                        "subject_id": str(user.id),
                        "role": user.role,
                        "public": user_to_public(user),
                    }
        except Exception as e:
            log.exception("Failed to establish fingerprint identity (%s). Headers: %s", e, {
                "X-Forwarded-For": request.headers.get("X-Forwarded-For"),
                "X-Client-Fingerprint": request.headers.get("X-Client-Fingerprint"),
                "User-Agent": request.headers.get("User-Agent"),
            })
            return None

    return None


def get_usage(subject_type: str, subject_id: str, period_type: str, period_key: str) -> int:
    with SessionLocal() as db:
        row = db.get(UsageCounter, (subject_type, subject_id, period_type, period_key))
        return int(row.count) if row else 0


def quota_snapshot(identity: Dict) -> Dict:
    limits = CONFIG["limits"][identity["role"]]
    daily_used = get_usage(identity["subject_type"], identity["subject_id"], "day", today_key())
    minute_used = get_usage(identity["subject_type"], identity["subject_id"], "minute", minute_key())
    return {
        "daily_limit": limits["daily"],
        "daily_used": daily_used,
        "daily_remaining": max(limits["daily"] - daily_used, 0),
        "minute_limit": limits["minute"],
        "minute_used": minute_used,
        "minute_remaining": max(limits["minute"] - minute_used, 0),
    }


def check_quota(identity: Dict) -> Tuple[bool, Dict, str]:
    usage = quota_snapshot(identity)
    if usage["daily_remaining"] <= 0:
        return False, usage, "Günlük soru limitin doldu."
    if usage["minute_remaining"] <= 0:
        return False, usage, "Dakikalık soru limitine ulaştın. Biraz bekleyip tekrar dene."
    return True, usage, ""


def increment_usage(identity: Dict) -> Dict:
    now = utc_now().isoformat()
    rows = [("day", today_key()), ("minute", minute_key())]
    with SessionLocal() as db:
        for period_type, period_key in rows:
            key = (
                identity["subject_type"],
                identity["subject_id"],
                period_type,
                period_key,
            )
            counter = db.get(UsageCounter, key)
            if counter is None:
                counter = UsageCounter(
                    subject_type=identity["subject_type"],
                    subject_id=identity["subject_id"],
                    period_type=period_type,
                    period_key=period_key,
                    count=1,
                    updated_at=now,
                )
                db.add(counter)
            else:
                counter.count += 1
                counter.updated_at = now
        db.commit()
    return quota_snapshot(identity)


@app.before_request
def enforce_https():
    if request.path.startswith("/api/health"):
        return None
    if CONFIG["force_https"] and not request.is_secure:
        host = request.headers.get("Host", "")
        is_local = host.startswith("127.0.0.1") or host.startswith("localhost")
        if not is_local:
            return "", 308, {"Location": request.url.replace("http://", "https://", 1)}
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Vector Store
# ═══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """Manages the FAISS vector database and chunk metadata."""

    def __init__(self, index_path: str, chunks_path: str, model_name: str):
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                "Run '01_build_vectorstore.py' first."
            )

        log.info("Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        log.info(f"FAISS index loaded: {self.index.ntotal} vectors")

        log.info("Loading chunk metadata...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[Dict] = json.load(f)
        log.info(f"Chunk metadata loaded: {len(self.chunks)} chunks")

        self.embedding_model_name = model_name
        self.gemini_keys = CONFIG["gemini_api_keys"]

        # Test embedding on startup
        log.info(f"Testing embedding API with {len(self.gemini_keys)} key(s)...")
        try:
            test_query = "test"
            embedding = self._embed_text(test_query)
            if embedding is not None:
                log.info(f"✓ Embedding API test successful (dim={embedding.shape})")
            else:
                log.error("✗ Embedding API test failed: returned None")
        except Exception as e:
            log.error(f"✗ Embedding API test failed: {e}")
            raise RuntimeError(f"Embedding service unavailable at startup: {e}")

        log.info(f"Vector store ready — {self.index.ntotal} chunks")

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Helper: embed a single text string.
        Returns normalized embedding array or None on failure.
        """
        for key_index, api_key in enumerate(self.gemini_keys, 1):
            if not api_key or not api_key.strip():
                log.debug(f"Gemini key {key_index} is empty, skipping")
                continue
            
            try:
                log.debug(f"Embedding with key {key_index}/{len(self.gemini_keys)}")
                client = genai.Client(api_key=api_key)
                response = client.models.embed_content(
                    model=self.embedding_model_name,
                    contents=text,
                    config={"task_type": "RETRIEVAL_QUERY"},
                )
                
                if not response.embeddings or not response.embeddings[0].values:
                    log.warning(f"Empty embedding response from key {key_index}")
                    continue
                
                embedding = np.array(
                    [response.embeddings[0].values],
                    dtype="float32"
                )
                faiss.normalize_L2(embedding)
                log.debug(f"Embedding successful with key {key_index}")
                return embedding
                
            except Exception as e:
                log.warning(f"Embedding failed with key {key_index}: {e}")
                continue
        
        return None

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Returns the top-k most relevant chunks for the given query.
        E5 model requires the 'query:' prefix for retrieval queries.
        """
        query_text = f"query: {query}"

        embedding = self._embed_text(query_text)
        if embedding is None:
            log.error(f"Could not embed query (all Gemini keys failed): {query[:100]}")
            raise RuntimeError(
                "Sorgu embedding'i başarısız. Lütfen daha sonra tekrar deneyin."
            )

        scores, indices = self.index.search(embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:    # FAISS returns -1 for empty slots
                continue
            chunk = self.chunks[idx].copy()
            chunk["relevance_score"] = float(score)
            results.append(chunk)

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Context Formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_context(retrieved_chunks: List[Dict], score_threshold: float = 0.35) -> str:
    """
    Builds the context string injected into the LLM prompt.
    Chunks below score_threshold are discarded to reduce noise.
    """
    if not retrieved_chunks:
        return "Bağlamda ilgili bilgi bulunamadı."

    parts = []
    for chunk in retrieved_chunks:
        if chunk.get("relevance_score", 0) < score_threshold:
            continue
        breadcrumb = chunk.get("breadcrumb", "")
        text = chunk.get("text", "")
        parts.append(f"[Kaynak: {breadcrumb}]\n{text}")

    return "\n\n---\n\n".join(parts) if parts else "Bağlamda yeterince ilgili bilgi bulunamadı."


def build_augmented_user_message(user_input: str, context: str) -> str:
    """Wraps the user question with the retrieved RAG context."""
    return (
        f"## İlgili Bağlam (Okul Bilgi Kaynağı)\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"## Kullanıcı Sorusu\n\n{user_input}"
    )


def build_sources_payload(retrieved: List[Dict], score_threshold: float = 0.35) -> List[Dict]:
    """Builds the sources list sent to the frontend after streaming ends."""
    return [
        {
            "breadcrumb": r.get("breadcrumb", ""),
            "score": round(r.get("relevance_score", 0), 3),
        }
        for r in retrieved[:3]
        if r.get("relevance_score", 0) >= score_threshold
    ]


def strip_reasoning_blocks(text: str) -> str:
    """Removes reasoning traces emitted by models that expose <think> blocks."""
    if not text:
        return text

    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<thinking\b[^>]*>.*?</thinking>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think\b[^>]*>.*\Z", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<thinking\b[^>]*>.*\Z", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Groq Backend (streaming)
# ═══════════════════════════════════════════════════════════════════════════════

def stream_groq_model(messages: List[Dict], model: str, api_key: str, key_index: int) -> Tuple[str, Optional[Dict]]:
    """
    Streams one Groq model attempt using curl_cffi to bypass Cloudflare 403 blocks on Render.
    Returns (full_response, failure_info).
    """
    log.warning("ENTERED stream_groq_model")
    full_response = ""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": CONFIG["llm_temperature"],
        "max_tokens": CONFIG["llm_max_tokens"],
        "top_p": CONFIG["llm_top_p"],
    }

    try:
        log.warning(
            "GROQ REQUEST -> model=%s key_index=%s url=%s",
            model,
            key_index,
            CONFIG["groq_url"]
        )
        # The impersonate="chrome" flag fully masks the TLS signature of the Render server,
        # making the handshake appear as a real Windows Chrome browser to Cloudflare.
        resp = curl_requests.post(
            CONFIG["groq_url"],
            headers=headers,
            json=payload,
            stream=True,
            timeout=CONFIG["groq_timeout"],
            impersonate="chrome"
        )

        log.warning(
            "GROQ RESPONSE <- status=%s headers=%s",
            resp.status_code,
            dict(resp.headers)
        )
        
        # HTTP error handling (catches 403 or other HTTP errors here)
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue

            # curl_cffi lines may return as string or bytes; safely decode them
            if isinstance(line, bytes):
                line_str = line.decode("utf-8")
            else:
                line_str = str(line)

            if not line_str.startswith("data: "):
                continue

            data_text = line_str[6:].strip()
            if data_text == "[DONE]":
                break

            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                continue

            delta = data.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"

    except curl_requests.errors.RequestsError as e:
        # Catch network/connection errors from the curl_cffi library
        error_msg = str(e).lower()
        reason = "exception"
        
        if "timeout" in error_msg:
            reason = "timeout"
            return "Groq API zaman aşımına uğradı. Lütfen tekrar deneyin.", {
                "retryable": True, "model": model, "key_index": key_index, "reason": reason
            }
        elif "connect" in error_msg or "resolve" in error_msg:
            reason = "connection"
            return "Groq API bağlantısı kurulamadı. Lütfen daha sonra tekrar deneyin.", {
                "retryable": True, "model": model, "key_index": key_index, "reason": reason
            }
            
        # If an HTTP status code is present (e.g., an HTTP error still came through)
        status_code = 0
        response_text = ""
        rate_headers = {}
        
        if hasattr(e, "response") and e.response is not None:
            status_code = e.response.status_code
            response_text = e.response.text
            log.error(
                "FULL GROQ BODY:\n%s",
                response_text
            )
            # Preserve the rate-limit header filtering logic
            rate_headers = {
                key: value
                for key, value in e.response.headers.items()
                if key.lower().startswith(("x-ratelimit", "retry-after", "x-request-id"))
            }
            reason = f"http_{status_code}"
            
        log.error(
            "Groq API HTTP error status=%s model=%s rate_headers=%s body=%s",
            status_code,
            model,
            rate_headers,
            response_text,
            exc_info=True,
        )
        
        if status_code > 0:
            retryable = status_code in {404, 429} or 500 <= status_code <= 599
            return f"Groq API hatası: HTTP {status_code}", {
                "retryable": retryable,
                "model": model,
                "key_index": key_index,
                "reason": reason,
                "status_code": status_code,
                "rate_headers": rate_headers,
            }
        else:
            return f"Groq API hatası: {str(e)}", {
                "retryable": True, "model": model, "key_index": key_index, "reason": reason
            }

    except Exception as e:
        log.exception("Groq streaming error model=%s", model)
        return f"Groq API hatası: {str(e)}", {
            "retryable": True,
            "model": model,
            "key_index": key_index,
            "reason": "exception",
        }

    return full_response, None


def stream_groq(messages: List[Dict]) -> Generator[str, None, None]:
    """
    Streams tokens from Groq's OpenAI-compatible Chat Completions API.
    Tries the full model chain for one API key, then rotates to the next key
    and starts from the strongest model again.
    """
    api_keys = CONFIG["groq_api_keys"]
    model_chain = CONFIG["groq_model_chain"]
    last_error = "Groq API hatası."

    if not api_keys:
        yield f"data: {json.dumps({'error': 'GROQ_API_KEY ayarlı değil.'})}\n\n"
        return

    for key_index, api_key in enumerate(api_keys, 1):
        for model_index, model in enumerate(model_chain):
            response_text = ""
            failure_info = None

            attempt = stream_groq_model(messages, model, api_key, key_index)
            while True:
                try:
                    yield next(attempt)
                except StopIteration as stop:
                    if stop.value:
                        response_text, failure_info = stop.value
                    break

            if failure_info is None:
                if model_index > 0:
                    notice = {
                        "from_model": model_chain[0],
                        "to_model": model,
                        "message": "Yoğunluk nedeniyle model düşürüldü.",
                    }
                    yield f"data: {json.dumps({'model_fallback': notice})}\n\n"
                    log.warning(
                        "Groq fallback succeeded key_index=%s original_model=%s active_model=%s",
                        key_index,
                        model_chain[0],
                        model,
                    )
                yield f"data: {json.dumps({'__full_response__': strip_reasoning_blocks(response_text)})}\n\n"
                return

            last_error = response_text
            if not failure_info.get("retryable"):
                yield f"data: {json.dumps({'error': last_error})}\n\n"
                return

            if model_index < len(model_chain) - 1:
                next_model = model_chain[model_index + 1]
                log.warning(
                    "Groq fallback switching key_index=%s from_model=%s to_model=%s reason=%s",
                    key_index,
                    model,
                    next_model,
                    failure_info.get("reason"),
                )
                continue

            if key_index < len(api_keys):
                log.warning(
                    "Groq API key exhausted key_index=%s next_key_index=%s last_model=%s reason=%s",
                    key_index,
                    key_index + 1,
                    model,
                    failure_info.get("reason"),
                )
                break

    yield f"data: {json.dumps({'error': last_error})}\n\n"
    return


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LLM Gateway
# ═══════════════════════════════════════════════════════════════════════════════

class LLMGateway:
    """
    Single backend-facing entry point for all model calls.
    Later this is where quotas, model routing and retries belong.
    """

    def __init__(self, config: Dict):
        self.config = config

    @property
    def active_provider(self) -> str:
        return self.config["provider"]

    def status(self) -> Dict:
        """Returns provider readiness for /api/health."""
        return {
            "provider": self.active_provider,
            "groq": bool(self.config["groq_api_keys"]),
            "groq_key_count": len(self.config["groq_api_keys"]),
            "model_name": self.config["groq_model_chain"][0],
            "model_chain": self.config["groq_model_chain"],
            "status": "ok" if self.config["groq_api_keys"] else "degraded",
        }

    def stream_chat(
        self,
        recent_history: List[Dict],
        augmented_message: str,
    ) -> Generator[str, None, None]:
        """
        Routes one chat turn to Groq.
        The Flask route does not call any external API directly.
        """
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + recent_history
            + [{"role": "user", "content": augmented_message}]
        )
        yield from stream_groq(messages)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Flask Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serves the frontend HTML file."""
    return send_from_directory(WEB_DIR, "index.html")

@app.route("/<path:filename>")
def serve_files(filename):
    return send_from_directory(WEB_DIR, filename)


@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    identity = get_current_identity()
    if not identity:
        return jsonify({
            "authenticated": False,
            "google_configured": bool(CONFIG["google_client_id"]),
            "google_client_id": CONFIG["google_client_id"],
            "https_required": CONFIG["force_https"],
        })

    usage = quota_snapshot(identity)
    limits = CONFIG["limits"].get(identity["role"], CONFIG["limits"]["user"])
    return jsonify({
        "authenticated": True,
        "user": identity["public"],
        "role": identity["role"],
        "daily_used": usage["daily_used"],
        "minute_used": usage["minute_used"],
        "daily_limit": limits["daily"],
        "minute_limit": limits["minute"],
        "near_limit": usage["daily_used"] >= 30,
        "google_configured": bool(CONFIG["google_client_id"]),
        "google_client_id": CONFIG["google_client_id"],
        "https_required": CONFIG["force_https"],
    })


def unsupported_auth():
    return jsonify({"error": "Authentication flow is not supported for this app."}), 404


@app.route("/api/auth/guest", methods=["POST"])
def auth_guest():
    return unsupported_auth()


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    return unsupported_auth()


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    return unsupported_auth()


@app.route("/api/auth/google", methods=["POST"])
def auth_google():
    return unsupported_auth()


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    return unsupported_auth()


@app.route("/api/health", methods=["GET"])
def health():
    """
    Returns a JSON status object.
    The LLM gateway reports provider readiness fields.
    """
    status = {
        "provider": CONFIG["provider"],
        "vectorstore": vector_store is not None,
        "database": database_ready(),
        "chunks": vector_store.index.ntotal if vector_store else 0,
    }

    if llm_gateway is None:
        status.update({"status": "degraded", "provider": None})
        return jsonify(status)

    provider_status = llm_gateway.status()
    status.update(provider_status)

    if not vector_store or not status["database"]:
        status["status"] = "degraded"

    return jsonify(status)


@app.route("/api/chat/feedback", methods=["POST"])
def chat_feedback():
    """
    Receives user feedback on a chat response.
    Request body (JSON):
        {
            "question_index": int,      # which question this feedback is for
            "feedback": "like" | "dislike" | None,  # vote
            "feedback_text": str | None  # optional written feedback
        }
    """
    body = request.get_json()
    if not body or "question_index" not in body:
        return jsonify({"error": "question_index gerekli"}), 400

    identity = get_current_identity()
    if not identity:
        return jsonify({"error": "Kimlik alınamadı"}), 401

    question_index = body["question_index"]
    feedback = body.get("feedback")
    feedback_text = body.get("feedback_text", "").strip()

    if feedback is not None and feedback not in ("like", "dislike"):
        return jsonify({"error": "feedback sadece 'like' veya 'dislike' olabilir"}), 400

    user_id = int(identity["subject_id"])
    try:
        with SessionLocal() as db:
            log_entry = db.query(ChatLog).filter(
                ChatLog.user_id == user_id,
                ChatLog.question_index == question_index,
            ).first()
            if not log_entry:
                return jsonify({"error": "Soru bulunamadı"}), 404

            if feedback is not None:
                log_entry.feedback = feedback
            if feedback_text:
                log_entry.feedback = "feedback"
                log_entry.feedback_text = feedback_text
            db.commit()
        return jsonify({"ok": True})
    except Exception as e:
        log.exception("Feedback save failed for user_id=%s question_index=%s", user_id, question_index)
        return jsonify({"error": "Geri bildirim kaydedilemedi"}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint — Server-Sent Events (SSE) streaming.

    Request body (JSON):
        {
            "message":    str,   # user's question
            "session_id": str    # optional, defaults to "default"
        }

    SSE event types streamed back:
        data: {"token": "..."}              — partial response token
        data: {"error": "..."}              — error message
        data: {"__full_response__": "..."}  — internal marker (consumed here)
        data: {"done": true, "sources": [...]} — final event with RAG sources
    """
    body = request.get_json()
    if not body or not body.get("message"):
        return jsonify({"error": "message alanı gerekli", "error_type": "technical"}), 400

    user_message = body["message"].strip()
    session_id = body.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Boş mesaj", "error_type": "technical"}), 400

    identity = get_current_identity()
    if not identity:
        try:
            headers_snapshot = {
                "X-Client-Fingerprint": request.headers.get("X-Client-Fingerprint"),
                "User-Agent": request.headers.get("User-Agent"),
                "Accept-Language": request.headers.get("Accept-Language"),
                "X-Forwarded-For": request.headers.get("X-Forwarded-For"),
            }
            log.warning("Visitor identity missing for /api/chat. Request cookies: %s, headers: %s, remote_addr: %s",
                        dict(request.cookies), headers_snapshot, request.remote_addr)
        except Exception:
            log.exception("Failed to log missing identity details")
        return jsonify({"error": "Ziyaretçi kimliği alınamadı; lütfen sayfayı yenileyin.", "error_type": "technical"}), 401

    # 🔥 LOGGING: Variables (identity and session_id) are now defined, safe to log.
    log.warning(
        "CHAT REQUEST user=%s session=%s msg=%s",
        identity["subject_id"] if identity else "unknown",
        session_id,
        user_message[:200]
    )

    quota_ok, quota, quota_error = check_quota(identity)
    if not quota_ok:
        return jsonify({"error": quota_error, "error_type": "quota"}), 429

    quota = increment_usage(identity)

    # Initialise session if new
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []

    history = conversation_sessions[session_id]

    # ── RAG: retrieve ONCE — result is reused for both prompt and sources ──────
    retrieved = vector_store.retrieve(user_message, top_k=CONFIG["retrieval_top_k"])
    context = format_context(retrieved, CONFIG["retrieval_score_threshold"])
    augmented_message = build_augmented_user_message(user_message, context)

    # Trim history to max_history_turns before building the prompt
    recent_history = history[-(CONFIG["max_history_turns"] * 2):]

    def generate():
        """
        Inner generator that drives the SSE stream.
        Intercepts the __full_response__ marker to persist history,
        then emits the final 'done' event with source metadata.
        """
        global active_requests
        full_response = ""
        had_error = False

        # ── Increment active request counter ─────────────────────────────────
        with active_requests_lock:
            active_requests += 1
            current_active = active_requests
            log.warning("CONGESTION active_requests=%s threshold=%s", current_active, CONGESTION_THRESHOLD)

        try:
            # ── Send congestion warning if threshold met or exceeded ────────
            if current_active >= CONGESTION_THRESHOLD:
                yield f"data: {json.dumps({'congestion': True, 'active_requests': current_active})}\n\n"

            # ── Send the turn through our backend gateway ────────────────────
            token_stream = llm_gateway.stream_chat(recent_history, augmented_message)

            # ── Forward tokens to the client, capture the full response ──────
            for event in token_stream:
                # Parse every event to check for internal markers
                if "__full_response__" in event:
                    try:
                        payload = json.loads(event.replace("data: ", "").strip())
                        full_response = payload.get("__full_response__", "")
                    except Exception:
                        pass
                    # Do NOT forward this internal marker to the client
                    continue

                if '"error"' in event:
                    had_error = True

                yield event   # Forward token or error events straight to the client

        except Exception:
            log.exception("Unexpected error during stream generation")
        finally:
            # ── Decrement active request counter ─────────────────────────────
            with active_requests_lock:
                active_requests -= 1
                log.warning("CONGESTION active_requests decremented to %s", active_requests)

        # ── Persist history (only on success) ────────────────────────────────
        if full_response and not had_error:
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": full_response})
            # Cap history length
            if len(history) > CONFIG["max_history_turns"] * 2:
                conversation_sessions[session_id] = history[-(CONFIG["max_history_turns"] * 2):]

            # Persist question-answer pair in chat_logs
            saved_question_index = None
            try:
                with SessionLocal() as db:
                    last_index = db.query(func.max(ChatLog.question_index)).filter(ChatLog.user_id == int(identity["subject_id"])).scalar()
                    saved_question_index = (last_index or 0) + 1
                    log_entry = ChatLog(
                        user_id=int(identity["subject_id"]),
                        question_index=saved_question_index,
                        question=user_message,
                        answer=full_response,
                        created_at=utc_now().isoformat(),
                    )
                    db.add(log_entry)
                    db.commit()
            except Exception:
                log.exception("Failed to save chat log for user_id=%s", identity["subject_id"])

        # ── Final event: sources ──────────────────────────────────────────────
        sources = build_sources_payload(retrieved, CONFIG["retrieval_score_threshold"])
        done_payload = {
            'done': True,
            'sources': sources,
            'near_limit': quota['daily_used'] >= 30,
        }
        if saved_question_index:
            done_payload['question_index'] = saved_question_index
        yield f"data: {json.dumps(done_payload)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Startup
# ═══════════════════════════════════════════════════════════════════════════════

def startup():
    """
    Runs once before the Flask server accepts requests.
    Loads the vector store and validates Groq configuration.
    """
    global vector_store, llm_gateway

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    log.info("BAL Chatbot Web API starting...")
    log.info(f"Runtime pid={os.getpid()} cwd={Path.cwd()} log_file={LOG_DIR / 'web.log'}")
    log.info(f"Provider: {CONFIG['provider']}")
    log.info(f"HTTPS enforcement: {CONFIG['force_https']}")
    if not CONFIG["secret_key"]:
        log.warning("FLASK_SECRET_KEY is not set. Sessions will reset after server restart.")

    # ── Load vector store ─────────────────────────────────────────────────────
    try:
        vector_store = VectorStore(
            CONFIG["faiss_index_file"],
            CONFIG["chunks_meta_file"],
            CONFIG["embedding_model"],
        )
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    # ── Groq configuration check ─────────────────────────────────────────────
    if not CONFIG["groq_api_keys"]:
        log.error(
            "No Groq API key is set. "
            "Set GROQ_API_KEY, GROQ_API_KEYS or GROQ_API_KEY_1..5 before starting the server."
        )
        sys.exit(1)
    log.info(
        "Groq configured — key_count=%s primary_model=%s",
        len(CONFIG["groq_api_keys"]),
        CONFIG["groq_model_chain"][0],
    )
    if not CONFIG["gemini_api_keys"]:
        log.error(
            "No Gemini API key configured."
        )
        sys.exit(1)

    llm_gateway = LLMGateway(CONFIG)
    log.info(f"LLM gateway ready — active provider: {llm_gateway.active_provider}")
    port = int(os.getenv("PORT", "5000"))
    scheme = "https" if CONFIG["local_https"] and not os.getenv("PORT") else "http"
    log.info(f"Server starting on {scheme}://0.0.0.0:{port}")

def run_startup_safely():
    global vector_store, llm_gateway
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # init_db is now defined above, so Python will recognize it and won't raise NameError!
    init_db()
    
    try:
        vector_store = VectorStore(
            CONFIG["faiss_index_file"],
            CONFIG["chunks_meta_file"],
            CONFIG["embedding_model"],
        )
    except Exception as e:
        print(f"CRITICAL: Vector store yuklenemedi: {e}")
        sys.exit(1)

    if not CONFIG["groq_api_keys"] or not CONFIG["gemini_api_keys"]:
        print("CRITICAL: API anahtarlari eksik!!!")
        sys.exit(1)

    llm_gateway = LLMGateway(CONFIG)

# CALLED HERE SO GUNICORN CAN START THE APPLICATION
#run_startup_safely()

# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    startup()
    port = int(os.getenv("PORT", "7860"))
    ssl_context = "adhoc" if CONFIG["local_https"] and not os.getenv("PORT") else None
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
        ssl_context=ssl_context,
    )
