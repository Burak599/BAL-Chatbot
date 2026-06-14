"""
=============================================================
 BAL Chatbot — Step 2: Chat Engine (RAG + LLM)
 Usage: python scripts/02_chatbot.py
=============================================================
This script:
  1. Uses Groq as the only LLM provider
  2. Loads FAISS index and chunk metadata
  3. Converts the user question into an embedding
  4. Retrieves the most relevant chunks (retrieval — done ONCE per query)
  5. Sends the augmented prompt to Groq
  6. Displays the response in the terminal
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
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_groq_api_key() -> str:
    """Read the Groq API key from the project .env/environment."""
    return os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_Key") or ""

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/chatbot.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # Vector database paths
    "faiss_index_file": str(PROJECT_ROOT / "data" / "bal_faiss.index"),
    "chunks_meta_file": str(PROJECT_ROOT / "data" / "bal_chunks.json"),

    # Embedding model (MUST match 01_build_vectorstore.py)
    "embedding_model": "intfloat/multilingual-e5-large",

    # How many chunks to retrieve per query (top-k)
    "retrieval_top_k": 5,

    # Minimum relevance score threshold — chunks below this are discarded
    "retrieval_score_threshold": 0.35,

    # ── Groq backend settings ────────────────────────────────────────────────
    "groq_url": "https://api.groq.com/openai/v1/chat/completions",
    "groq_model": "llama-3.3-70b-versatile",
    "groq_api_key": get_groq_api_key(),
    "groq_timeout": 120,             # seconds

    # ── LLM generation parameters ────────────────────────────────────────────
    "llm_temperature": 0.3,          # lower = more consistent
    "llm_max_tokens": 1024,
    "llm_top_p": 0.9,

    # Conversation history — how many previous turns to keep in context
    "max_history_turns": 6,
}

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Sen BAL Asistan'sın — Bornova Anadolu Lisesi'nin yapay zeka asistanı. BAL Yapay Zeka Topluluğu tarafından geliştirildin.

## GÖREV
Öğrencilere, velilere ve meraklılara BAL hakkında doğru, kısa ve samimi bilgi vermek.

## TON VE ÜSLUP
- Kısa ve net konuş. Dolgu cümlesi yok: "Umarım yardımcı olur", "sormaktan çekinmeyin", "tabii ki" gibi kalıpları kullanma.
- Samimi ve doğal ol — ne aşırı resmi ne aşırı neşeli.
- Gerekmedikçe liste yapma; soruyu doğrudan yanıtla.
- Selamlama, teşekkür, vedaya zaman harcama — direkt konuya gir.

- Telefon numarası, URL gibi somut verileri ASLA değiştirme veya uydurma. 
  Bağlamda yazan bilgiyi olduğu gibi kullan.
- Türkçe yaz, İngilizce kelime karıştırma.

## BİLGİ KAPSAMI
Yalnızca şu konularda bilgi ver:
- Okul tarihi, bölümler, eğitim yapısı
- LGS taban puanları ve yerleştirme
- Kampüs olanakları (laboratuvar, spor salonu, kütüphane, pansiyon vb.)
- Okul kültürü (BAL Ruhu, Ayran Günü, marş, müzik geleneği)
- Kulüpler ve topluluklar (tiyatro, fotoğraf, BAL Radyo, BALspor, Ultimate Frizbi vb.)
- Uluslararası programlar (PASCH, eTwinning, DSD, AP)
- BALEV bursları, BALMED, Bi'BALlı mentorlük
- Kayıt, nakil, devamsızlık, pansiyon
- Ulaşım ve iletişim bilgileri

## KAYNAK KULLANIMI
Verilen bağlam (RAG) birincil kaynağın. Bağlamda varsa oradan cevap ver. Bağlamda yoksa şunu söyle: "Bu konuda kesin bilgim yok, okul idaresiyle teyit etmeni öneririm." — Asla uydurma.

## SINIRLAR
- Okul dışı konular (politika, genel haberler, kişisel tavsiye vb.): "Bu konuda yardımcı olamam, BAL hakkında bir sorun var mı?" de ve geç.
- Bireysel öğrenci verisi (not, devamsızlık durumu, sınıf listesi): "Bu bilgilere erişimim yok, okul idaresiyle iletişime geç." de.
- "Seni kim yaptı / sen ne düşünüyorsun / sen kimsin": BAL Yapay Zeka Topluluğu tarafından geliştirildiğini söyle, fazla uzatma.

## ASLA YAZMA:
- "bağlamı kontrol etmem gerekiyor"
- "bağlamda bilgi var/yok"
- "bağlamı inceliyorum"
- "soruyu cevaplamak için"
- "umarım yardımcı olur"
- "sormaktan çekinmeyin"
- Direkt cevap ver. Bu kadar.

## ÖZEL DURUMLAR
- Hakaret veya uygunsuz dil: Tek cümleyle kibarca uyar ve konuya dön.
- Belirsiz soru: Ne sorduğunu tek cümleyle sor.
- Bilgi bağlamda işaretliyse güncel olmayabilir: "Kesin bilgi için okul idaresiyle teyit et" ekini koy — ama bunu her cevaba yapıştırma, sadece gerçekten gerektiğinde yaz.

## YARDIMCI LİNKLER (yalnızca sorulduğunda ya da doğrudan ilgiliyse ver)
- Okul sitesi: izmirbal.meb.k12.tr
- BALEV: balev.org.tr
- BALMED: balmed.org.tr
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Vector Store
# ═══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """Manages the FAISS vector database and chunk metadata."""

    def __init__(self, index_path: str, chunks_path: str, model_name: str):
        # Load FAISS index
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                "Run '01_build_vectorstore.py' first."
            )
        self.index = faiss.read_index(index_path)

        # Load chunk metadata
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[Dict] = json.load(f)

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        print(f"  ✓ Vector store loaded ({self.index.ntotal} chunks)")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Returns the top-k most relevant chunks for the given query.
        E5 model requires the 'query:' prefix for queries.
        """
        query_text = f"query: {query}"
        embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:    # FAISS sometimes returns -1 for empty slots
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
    Builds the context string that is injected into the LLM prompt.
    Chunks below the score threshold are skipped to reduce noise.
    """
    if not retrieved_chunks:
        return "Bağlamda ilgili bilgi bulunamadı."

    context_parts = []
    for chunk in retrieved_chunks:
        score = chunk.get("relevance_score", 0)
        if score < score_threshold:
            log.debug(f"Low-score chunk skipped: score={score:.3f}")
            continue
        breadcrumb = chunk.get("breadcrumb", "")
        text = chunk.get("text", "")
        context_parts.append(f"[Kaynak: {breadcrumb}]\n{text}")

    if not context_parts:
        return "Bağlamda yeterince ilgili bilgi bulunamadı."

    return "\n\n---\n\n".join(context_parts)


def build_augmented_user_message(user_input: str, context: str) -> str:
    """Wraps user input with the retrieved RAG context."""
    return (
        f"## İlgili Bağlam (Okul Bilgi Kaynağı)\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"## Kullanıcı Sorusu\n\n{user_input}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Groq Backend
# ═══════════════════════════════════════════════════════════════════════════════

def query_groq(messages: List[Dict], config: Dict) -> str:
    """
    Sends a streaming chat request to Groq and prints tokens as they arrive.
    Returns the full response text.
    """
    headers = {
        "Authorization": f"Bearer {config['groq_api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["groq_model"],
        "messages": messages,
        "stream": True,
        "temperature": config["llm_temperature"],
        "max_tokens": config["llm_max_tokens"],
        "top_p": config["llm_top_p"],
    }

    full_response = ""
    try:
        with requests.post(
            config["groq_url"],
            headers=headers,
            json=payload,
            stream=True,
            timeout=config["groq_timeout"],
        ) as resp:
            resp.raise_for_status()
            print("\n\033[94mBAL Asistan:\033[0m ", end="", flush=True)

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8")
                if not line.startswith("data: "):
                    continue

                data_text = line[6:].strip()
                if data_text == "[DONE]":
                    print()
                    break

                try:
                    data = json.loads(data_text)
                except json.JSONDecodeError:
                    continue

                delta = data.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_response += token

    except requests.exceptions.ConnectionError:
        full_response = "Groq API bağlantısı kurulamadı. Lütfen daha sonra tekrar deneyin."
        print(f"\n\033[91m{full_response}\033[0m")
    except requests.exceptions.Timeout:
        full_response = "Groq API zaman aşımına uğradı. Lütfen tekrar deneyin."
        print(f"\n\033[91m{full_response}\033[0m")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "?"
        full_response = f"Groq API hatası: HTTP {status_code}"
        print(f"\n\033[91m{full_response}\033[0m")
        log.exception("Groq API HTTP error")
    except Exception as e:
        full_response = f"Groq API hatası: {e}"
        print(f"\n\033[91m{full_response}\033[0m")
        log.exception("Groq query error")

    return full_response


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Conversation Manager
# ═══════════════════════════════════════════════════════════════════════════════

class Conversation:
    """
    Manages conversation history and the full RAG → LLM pipeline.
    Retrieval is done ONCE per user query; the result is reused for
    both the LLM prompt and the /kaynak command.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        config: Dict,
    ):
        self.vs = vector_store
        self.config = config
        # Plain message history (no RAG context injected — keeps history compact)
        self.history: List[Dict] = []
        # Stores the last retrieved chunks so /kaynak can display them
        self.last_retrieved: List[Dict] = []

    def ask(self, user_input: str) -> str:
        """
        Full pipeline for one conversational turn:
          1. Retrieve relevant chunks (ONCE)
          2. Build context string
          3. Build augmented user message (context + question)
          4. Send to Groq
          5. Append plain texts to history (no duplicate RAG context)
        Returns the assistant's response text.
        """
        user_input = user_input.strip()

        # ── 1. Retrieve ───────────────────────────────────────────────────────
        t_ret = time.time()
        retrieved = self.vs.retrieve(user_input, top_k=self.config["retrieval_top_k"])
        self.last_retrieved = retrieved   # cache for /kaynak command
        log.debug(f"Retrieval: {len(retrieved)} chunks in {time.time() - t_ret:.2f}s")

        # ── 2. Build context ──────────────────────────────────────────────────
        context = format_context(retrieved, self.config["retrieval_score_threshold"])

        # ── 3. Build augmented message ────────────────────────────────────────
        augmented_message = build_augmented_user_message(user_input, context)

        # ── 4. Trim history to max_history_turns ──────────────────────────────
        recent_history = self.history[-(self.config["max_history_turns"] * 2):]

        # ── 5. Query Groq ────────────────────────────────────────────────────
        t_llm = time.time()
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + recent_history
            + [{"role": "user", "content": augmented_message}]
        )
        response = query_groq(messages, self.config)

        log.debug(f"LLM response: {len(response)} chars in {time.time() - t_llm:.2f}s")

        # ── 6. Store plain texts in history (no embedded context) ─────────────
        if response and not response.startswith(("Bir hata", "Groq API")):
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self):
        """Clears the conversation history."""
        self.history.clear()
        self.last_retrieved.clear()
        print("\n\033[93m[Conversation history cleared]\033[0m\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Terminal UI
# ═══════════════════════════════════════════════════════════════════════════════

CHAT_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                   RAG-Powered Chatbot                        ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                    ║
║    /temizle  → Clear conversation history                    ║
║    /kaynak   → Show sources from the last query              ║
║    /çıkış   → Exit the program                               ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_sources(retrieved: List[Dict]):
    """Prints the source breadcrumbs of the last retrieved chunks."""
    print("\n\033[93m── Sources Used ────────────────────────────────────\033[0m")
    if not retrieved:
        print("  (No question has been asked yet)")
    else:
        for i, chunk in enumerate(retrieved, 1):
            score = chunk.get("relevance_score", 0)
            breadcrumb = chunk.get("breadcrumb", "")
            words = chunk.get("word_count", 0)
            print(f"  {i}. [{score:.3f}] {breadcrumb} ({words} words)")
    print("\033[93m────────────────────────────────────────────────────\033[0m\n")


def run_cli():
    """Main command-line chat loop."""

    if not CONFIG["groq_api_key"]:
        print(
            "\n\033[91mGROQ_API_KEY is not set.\033[0m\n"
            "Set the API key in the terminal and run again:\n"
            "  \033[1mexport GROQ_API_KEY='...'\033[0m\n"
        )
        sys.exit(1)

    # ── Load vector store ─────────────────────────────────────────────────────
    print("\n\033[96mLoading vector database...\033[0m")
    try:
        vs = VectorStore(
            CONFIG["faiss_index_file"],
            CONFIG["chunks_meta_file"],
            CONFIG["embedding_model"],
        )
    except FileNotFoundError as e:
        print(f"\n\033[91m{e}\033[0m\n")
        sys.exit(1)

    # ── Start conversation ────────────────────────────────────────────────────
    conv = Conversation(vs, CONFIG)

    print(CHAT_BANNER)
    print(f"\033[92m✅ System ready!  Active model: Groq / {CONFIG['groq_model']}\033[0m\n")

    while True:
        try:
            user_input = input("\033[1mYou:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSee you later! 👋")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.lower() in ("/çıkış", "/cikis", "çıkış", "exit", "quit"):
            print("\nSee you later! 👋")
            break

        if user_input.lower() in ("/temizle", "/temizle"):
            conv.clear_history()
            continue

        if user_input.lower() == "/kaynak":
            print_sources(conv.last_retrieved)
            continue

        # ── Ask ───────────────────────────────────────────────────────────────
        print()
        conv.ask(user_input)
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_cli()