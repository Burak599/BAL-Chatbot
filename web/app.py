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
import hashlib

import numpy as np
import faiss
import requests
from sqlalchemy import Column, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker
from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory, session
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from google import genai

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
    "force_https": env_bool("FORCE_HTTPS", True),
    "local_https": env_bool("LOCAL_HTTPS", True),
    "google_client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
    "admin_emails": split_env_list("ADMIN_EMAILS"),
    "limits": {
        "visitor": {"daily": 40, "minute": 5},
        "user": {"daily": 50, "minute": 8},
        "admin": {"daily": 500, "minute": 20},
    },
}

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are BAL Asistan, the AI assistant of Bornova Anadolu Lisesi. You were developed by BAL Yapay Zeka Topluluğu.

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

## INFORMATION SCOPE
Only answer questions about:
- School history, departments and education structure
- LGS base scores and placement
- Campus facilities such as laboratories, sports halls, library, dormitory and cafeteria
- School culture such as BAL Ruhu, Ayran Günü, school anthem and music tradition
- Clubs and communities such as theatre, photography, BAL Radyo, BALspor and Ultimate Frizbi
- International programs such as PASCH, eTwinning, DSD and AP
- BALEV scholarships, BALMED and Bi'BALlı mentoring
- Registration, transfer, absenteeism and dormitory
- Transportation and contact information
- YKS and university achievements

## SOURCE USE
The provided RAG context is your primary source. If the answer is in the context, answer from it.
If the context does not contain enough reliable information, say exactly: "Bu konuda kesin bilgim yok, okul idaresiyle teyit etmeni öneririm."
Never make things up.

## BOUNDARIES
- For topics outside BAL, such as politics, general news or personal advice, say: "Bu konuda yardımcı olamam, BAL hakkında bir sorun var mı?"
- For individual student data such as grades, absenteeism status or class lists, say: "Bu bilgilere erişimim yok, okul idaresiyle iletişime geç."
- If asked who made you, what you think, or who you are, briefly say that you were developed by BAL Yapay Zeka Topluluğu.
- If the user insults you or uses inappropriate language, do not insult back and do not use profanity. Give one polite warning and return to the BAL topic.

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


def init_db() -> None:
    if CONFIG["database_url"].startswith("sqlite"):
        Path(PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _ensure_user_fingerprint_column()


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
    limits = CONFIG["limits"].get(role, CONFIG["limits"]["user"])
    # treat fingerprint-identified users as visitors for display
    is_visitor = role == "visitor" or user.provider == "fingerprint"
    return {
        "id": user.id,
        "email": None if is_visitor else user.email,
        "role": "visitor" if is_visitor else role,
        "mode": "visitor" if is_visitor else "account",
    }


def build_fingerprint() -> str:
    user_agent = request.headers.get("User-Agent", "")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr) or ""
    language = request.headers.get("Accept-Language", "")
    fingerprint_source = f"{ip}|{user_agent}|{language}"
    return hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()


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

    # 2) Fingerprint-based identity (preferred for anonymous flow)
    fingerprint = session.get("fingerprint")
    if not fingerprint:
        fingerprint = request.headers.get("X-Client-Fingerprint")
    if not fingerprint:
        fingerprint = build_fingerprint()

    if fingerprint:
        with SessionLocal() as db:
            user = db.query(User).filter(User.fingerprint == fingerprint).first()
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
                except IntegrityError:
                    db.rollback()
                    user = db.query(User).filter(User.fingerprint == fingerprint).first()

            if user:
                session["fingerprint"] = fingerprint
                return {
                    "subject_type": "user",
                    "subject_id": str(user.id),
                    "role": user.role,
                    "public": user_to_public(user),
                }

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

        log.info("Loading chunk metadata...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[Dict] = json.load(f)

        self.embedding_model_name = model_name
        self.gemini_keys = CONFIG["gemini_api_keys"]

        log.info(f"Vector store ready — {self.index.ntotal} chunks")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Returns the top-k most relevant chunks for the given query.
        E5 model requires the 'query:' prefix for retrieval queries.
        """
        query_text = f"query: {query}"

        embedding = None
        last_error = None

        for key_index, api_key in enumerate(self.gemini_keys,1):

            try:

                client = genai.Client(api_key=api_key)

                client = genai.Client(api_key=api_key)

                response = client.models.embed_content(
                    model="models/gemini-embedding-001",
                    contents=query_text,
                    config={
                        "task_type":"RETRIEVAL_QUERY"
                    }
                )

                embedding = np.array(
                    [response.embeddings[0].values],
                    dtype="float32"
                )

                faiss.normalize_L2(embedding)

                break

            except Exception as e:

                log.warning(
                    f"Gemini embedding failed key={key_index} error={e}"
                )

                last_error=e

                continue

        if embedding is None:
            raise RuntimeError(
                f"All Gemini embedding keys failed: {last_error}"
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
    Streams one Groq model attempt.
    Returns (full_response, failure_info). failure_info is set for retryable
    provider throttling/server errors after any already-yielded error event.
    """
    full_response = ""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
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
        with requests.post(
            CONFIG["groq_url"],
            headers=headers,
            json=payload,
            stream=True,
            timeout=CONFIG["groq_timeout"],
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8")
                if not line.startswith("data: "):
                    continue

                data_text = line[6:].strip()
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

    except requests.exceptions.ConnectionError:
        return "Groq API bağlantısı kurulamadı. Lütfen daha sonra tekrar deneyin.", {
            "retryable": True,
            "model": model,
            "key_index": key_index,
            "reason": "connection",
        }
    except requests.exceptions.Timeout:
        return "Groq API zaman aşımına uğradı. Lütfen tekrar deneyin.", {
            "retryable": True,
            "model": model,
            "key_index": key_index,
            "reason": "timeout",
        }
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 0
        response_text = ""
        rate_headers = {}
        if e.response is not None:
            response_text = e.response.text[:2000]
            rate_headers = {
                key: value
                for key, value in e.response.headers.items()
                if key.lower().startswith(("x-ratelimit", "retry-after", "x-request-id"))
            }
        log.error(
            "Groq API HTTP error status=%s model=%s rate_headers=%s body=%s",
            status_code,
            model,
            rate_headers,
            response_text,
            exc_info=True,
        )
        # 404 can happen when a Groq model is unavailable, deprecated, or not
        # enabled for the project. Keep walking the fallback chain.
        retryable = status_code in {404, 429} or 500 <= status_code <= 599
        return f"Groq API hatası: HTTP {status_code}", {
            "retryable": retryable,
            "model": model,
            "key_index": key_index,
            "reason": f"http_{status_code}",
            "status_code": status_code,
            "rate_headers": rate_headers,
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
    return jsonify({
        "authenticated": True,
        "user": identity["public"],
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
        return jsonify({"error": "message alanı gerekli"}), 400

    user_message = body["message"].strip()
    session_id = body.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Boş mesaj"}), 400

    identity = get_current_identity()
    if not identity:
        return jsonify({"error": "Ziyaretçi kimliği alınamadı; lütfen sayfayı yenileyin."}), 401

    quota_ok, quota, quota_error = check_quota(identity)
    if not quota_ok:
        return jsonify({"error": quota_error}), 429

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
        full_response = ""
        had_error = False

        # ── Send the turn through our backend gateway ────────────────────────
        token_stream = llm_gateway.stream_chat(recent_history, augmented_message)

        # ── Forward tokens to the client, capture the full response ───────────
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

        # ── Persist history (only on success) ────────────────────────────────
        if full_response and not had_error:
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": full_response})
            # Cap history length
            if len(history) > CONFIG["max_history_turns"] * 2:
                conversation_sessions[session_id] = history[-(CONFIG["max_history_turns"] * 2):]

        # ── Final event: sources ──────────────────────────────────────────────
        sources = build_sources_payload(retrieved, CONFIG["retrieval_score_threshold"])
        yield f"data: {json.dumps({'done': True, 'sources': sources, 'near_limit': quota['daily_used'] >= 30})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/clear", methods=["POST"])
def clear_session():
    """Clears the conversation history for the given session."""
    body = request.get_json() or {}
    session_id = body.get("session_id", "default")
    if session_id in conversation_sessions:
        conversation_sessions[session_id] = []
        log.info(f"Session cleared: {session_id}")
    return jsonify({"status": "cleared"})


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
    
    # Artık init_db yukarıda tanımlandığı için Python bunu tanıyacak ve NameError vermeyecek!
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
        print("CRITICAL: API anahtarlari eksik!")
        sys.exit(1)

    llm_gateway = LLMGateway(CONFIG)

# GUNICORN'UN UYGULAMAYI BAŞLATMASI İÇİN BURADA ÇAĞIRIYORUZ
run_startup_safely()

# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    ssl_context = "adhoc" if CONFIG["local_https"] and not os.getenv("PORT") else None
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
        ssl_context=ssl_context,
    )
