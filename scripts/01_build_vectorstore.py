"""
=============================================================
 BAL Chatbot — Step 1: Build Vector Database
 Usage: python scripts/01_build_vectorstore.py
=============================================================
This script:
  1. Reads the RAG_Dataset_BAL.md markdown file
  2. Splits markdown into semantically meaningful chunks
  3. Generates embeddings for each chunk via Gemini API
  4. Stores vectors in a FAISS index for similarity search
  5. Writes chunk metadata to JSON for fast retrieval
=============================================================
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from google import genai

import numpy as np
import faiss
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/build_vectorstore.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def get_gemini_api_keys():
    """Reads Gemini API keys from environment variables GEMINI_API_KEY_1..5."""
    keys = []
    for i in range(1, 6):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    return keys


# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # Path to the raw markdown dataset file
    "dataset_path": str(PROJECT_ROOT / "Dataset" / "RAG_Dataset_BAL.md"),
    # Embedding model — multilingual, strong Turkish support
    "embedding_model": "models/gemini-embedding-001",
    "gemini_api_keys": get_gemini_api_keys(),
    # Alternatives (faster but slightly lower quality):
    #   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    #   "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    "chunk_size": 400,           # Maximum chunk size in words
    "chunk_overlap": 80,         # Word overlap between consecutive chunks
    "output_dir": str(PROJECT_ROOT / "data"),
    "faiss_index_file": str(PROJECT_ROOT / "data" / "bal_faiss.index"),
    "chunks_meta_file": str(PROJECT_ROOT / "data" / "bal_chunks.json"),
    "vectorstore_config_file": str(PROJECT_ROOT / "data" / "vectorstore_config.json"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Document Loading & Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def load_markdown(path: str) -> str:
    """Reads a markdown file and performs basic text cleanup."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Ensure RAG_Dataset_BAL.md exists or update CONFIG['dataset_path']."
        )
    text = p.read_text(encoding="utf-8")
    log.info(f"File loaded: {path} ({len(text):,} chars)")
    return text


def extract_sections(markdown: str) -> List[Dict]:
    """
    Splits markdown into logical sections based on headers.
    Each section: {"title": str, "level": int, "content": str, "breadcrumb": str}
    """
    sections = []
    # Header pattern: ## Title, ### Subtitle, etc.
    header_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)

    # Find all header positions and content
    matches = list(header_pattern.finditer(markdown))

    breadcrumb_stack = {}  # level -> title

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # Update breadcrumb trail
        breadcrumb_stack[level] = title
        # Clear deeper levels
        for lvl in list(breadcrumb_stack.keys()):
            if lvl > level:
                del breadcrumb_stack[lvl]

        # Section content: from this header to the next one
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = markdown[start:end].strip()

        # Skip very short sections (header-only, empty sections)
        if len(content) < 30:
            continue

        breadcrumb = " > ".join(breadcrumb_stack.values())

        sections.append({
            "title": title,
            "level": level,
            "content": content,
            "breadcrumb": breadcrumb,
        })

    log.info(f"  {len(sections)} sections extracted")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Smart Chunking
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Strips markdown syntax and produces plain text."""
    # Merge table rows
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'^[-\s|]+$', '', text, flags=re.MULTILINE)
    # Markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    # Links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Heading markers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Multiple whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def split_into_chunks(
    text: str,
    max_words: int,
    overlap_words: int
) -> List[str]:
    """
    Splits text into word-based overlapping chunks.
    Respects sentence boundaries (periods, question marks, exclamation marks).
    """
    # First split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_words = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)

        if current_word_count + sentence_word_count > max_words and current_words:
            # Save current chunk
            chunks.append(" ".join(current_words))
            # Keep last N words for overlap
            overlap_start = max(0, len(current_words) - overlap_words)
            current_words = current_words[overlap_start:] + sentence_words
            current_word_count = len(current_words)
        else:
            current_words.extend(sentence_words)
            current_word_count += sentence_word_count

    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if len(c.strip()) > 50]


def build_chunks(sections: List[Dict], config: Dict) -> List[Dict]:
    """
    Splits each section into chunks and attaches rich metadata.
    """
    all_chunks = []
    chunk_id = 0

    for section in sections:
        clean = clean_text(section["content"])
        sub_chunks = split_into_chunks(
            clean,
            config["chunk_size"],
            config["chunk_overlap"]
        )

        for i, chunk_text in enumerate(sub_chunks):
            # Prepends breadcrumb for embedding context
            embed_text = f"{section['breadcrumb']}\n\n{chunk_text}"

            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text,               # Raw text (for display)
                "embed_text": embed_text,          # Text sent to embedding model
                "section_title": section["title"],
                "breadcrumb": section["breadcrumb"],
                "section_level": section["level"],
                "chunk_index_in_section": i,
                "total_chunks_in_section": len(sub_chunks),
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
            })
            chunk_id += 1

    log.info(f"  Total {len(all_chunks)} chunks created")
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Embedding Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_embeddings(chunks: List[Dict], model_name: str) -> np.ndarray:
    """
    Generates embedding vectors for every chunk using the Gemini API.
    Uses batch processing for large datasets and falls back across multiple API keys.
    """
    log.info(f"Loading embedding model: {model_name}")

    api_keys = CONFIG["gemini_api_keys"]

    if not api_keys:
        raise RuntimeError("No Gemini API keys configured.")

    texts = [c["embed_text"] for c in chunks]
    total = len(texts)

    log.info(f"Generating embeddings for {total} chunks...")

    batch_size = 32
    all_embeddings = []

    for i in range(0, total, batch_size):

        batch = texts[i: i + batch_size]
        success = False

        for key in api_keys:
            try:
                client = genai.Client(api_key=key)
                batch_embeddings = []

                for text in batch:
                    response = client.models.embed_content(
                        model=model_name,
                        contents=text,
                        config={"task_type": "RETRIEVAL_DOCUMENT"}
                    )
                    batch_embeddings.append(response.embeddings[0].values)

                embeddings = np.array(batch_embeddings, dtype="float32")

                # Normalize vectors for cosine similarity
                embeddings = embeddings / np.linalg.norm(
                    embeddings, axis=1, keepdims=True
                )

                success = True
                break

            except Exception as e:
                log.warning(f"API key failed: {str(e)}")
                continue

        if not success:
            raise RuntimeError("All Gemini API keys exhausted.")

        all_embeddings.append(embeddings)

        log.info(f"[{i + len(batch)}/{total}] chunks processed")

    result = np.vstack(all_embeddings).astype("float32")
    log.info(f"  Embedding shape: {result.shape}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FAISS Vector Database
# ═══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Creates a FAISS Inner Product (cosine) index.
    IndexFlatIP is the most reliable choice for small/medium datasets.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"  FAISS index created: {index.ntotal} vectors, dim={dim}")
    return index


def save_artifacts(
    index: faiss.IndexFlatIP,
    chunks: List[Dict],
    config: Dict
) -> None:
    """Saves the FAISS index and chunk metadata to disk."""
    os.makedirs(config["output_dir"], exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, config["faiss_index_file"])
    log.info(f"  FAISS index saved: {config['faiss_index_file']}")

    # Save chunk metadata (strip embed_text to save disk space)
    chunks_for_save = [
        {k: v for k, v in c.items() if k != "embed_text"}
        for c in chunks
    ]
    with open(config["chunks_meta_file"], "w", encoding="utf-8") as f:
        json.dump(chunks_for_save, f, ensure_ascii=False, indent=2)
    log.info(f"  Chunk metadata saved: {config['chunks_meta_file']}")

    # Save config snapshot (records which model and parameters were used)
    config_snapshot = {
        **config,
        "total_chunks": len(chunks),
        "embedding_dim": index.d,
        "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config["vectorstore_config_file"], "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)
    log.info(f"  Config snapshot saved: {config['vectorstore_config_file']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("BAL Chatbot — Vector Database Build Started")
    log.info("=" * 60)

    t0 = time.time()

    # 1. Load the document
    markdown = load_markdown(CONFIG["dataset_path"])

    # 2. Split into sections
    log.info("Splitting document into sections...")
    sections = extract_sections(markdown)

    # 3. Split into chunks
    log.info("Creating chunks...")
    chunks = build_chunks(sections, CONFIG)

    # Statistics
    word_counts = [c["word_count"] for c in chunks]
    log.info(
        f"  Chunk statistics — "
        f"min: {min(word_counts)}, "
        f"max: {max(word_counts)}, "
        f"avg: {sum(word_counts) / len(word_counts):.0f} words"
    )

    # 4. Generate embeddings
    embeddings = generate_embeddings(chunks, CONFIG["embedding_model"])

    # 5. Build FAISS index
    log.info("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # 6. Save artifacts
    log.info("Saving artifacts...")
    save_artifacts(index, chunks, CONFIG)

    elapsed = time.time() - t0
    log.info(f"\n✅ Complete! Elapsed: {elapsed:.1f}s")
    log.info(f"   Total chunks: {len(chunks)}")
    log.info(f"   FAISS index: {CONFIG['faiss_index_file']}")
    log.info(f"   Chunk metadata: {CONFIG['chunks_meta_file']}")
    log.info("\nNext step: python scripts/02_chatbot.py")


if __name__ == "__main__":
    main()