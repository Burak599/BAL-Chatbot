"""
=============================================================
 BAL Chatbot — Adım 1: Vektör Veritabanı Oluşturma
 Kullanım: python scripts/01_build_vectorstore.py
=============================================================
Bu script:
  1. RAG_Dataset_BAL.md dosyasını okur
  2. Markdown'ı anlamlı chunk'lara böler
  3. Her chunk için embedding üretir (sentence-transformers)
  4. FAISS vektör veritabanına kaydeder
  5. Chunk metadata'sını JSON'a yazar (hızlı erişim için)
=============================================================
"""

import os
import re
import json
import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/build_vectorstore.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Konfigürasyon ─────────────────────────────────────────────────────────────
CONFIG = {
    # Doküman yolu (masaüstünde olduğunu varsayıyoruz; gerekirse düzeltin)
    "dataset_path": str(PROJECT_ROOT / "Dataset" / "RAG_Dataset_BAL.md"),
    # Embedding modeli — çok dilli, Türkçe'yi iyi destekler
    "embedding_model": "intfloat/multilingual-e5-large",
    # Alternatifler (daha hızlı ama biraz daha düşük kalite):
    #   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    #   "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    "chunk_size": 400,         # Kelime bazlı max chunk boyutu
    "chunk_overlap": 80,       # Chunk'lar arası örtüşme (kelime)
    "output_dir": str(PROJECT_ROOT / "data"),
    "faiss_index_file": str(PROJECT_ROOT / "data" / "bal_faiss.index"),
    "chunks_meta_file": str(PROJECT_ROOT / "data" / "bal_chunks.json"),
    "vectorstore_config_file": str(PROJECT_ROOT / "data" / "vectorstore_config.json"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Doküman Okuma ve Ön İşleme
# ═══════════════════════════════════════════════════════════════════════════════

def load_markdown(path: str) -> str:
    """Markdown dosyasını okur ve temel temizlik yapar."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset bulunamadı: {path}\n"
            "Lütfen RAG_Dataset_BAL.md dosyasını Desktop klasörüne koyun "
            "veya CONFIG['dataset_path'] değerini güncelleyin."
        )
    text = p.read_text(encoding="utf-8")
    log.info(f"Dosya okundu: {path} ({len(text):,} karakter)")
    return text


def extract_sections(markdown: str) -> List[Dict]:
    """
    Markdown başlıklarına göre mantıksal bölümlere ayırır.
    Her bölüm: {"title": str, "level": int, "content": str, "breadcrumb": str}
    """
    sections = []
    # Başlık pattern'i: ## Başlık, ### Alt Başlık, vb.
    header_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    
    # Başlıkların konum ve içeriklerini bul
    matches = list(header_pattern.finditer(markdown))
    
    breadcrumb_stack = {}  # level -> title

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        
        # Breadcrumb güncelle
        breadcrumb_stack[level] = title
        # Daha derin seviyeleri temizle
        for lvl in list(breadcrumb_stack.keys()):
            if lvl > level:
                del breadcrumb_stack[lvl]
        
        # Bölüm içeriği: bu başlıktan bir sonraki başlığa kadar
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = markdown[start:end].strip()
        
        # Çok kısa bölümleri atla (başlık satırları, boş bölümler)
        if len(content) < 30:
            continue
        
        breadcrumb = " > ".join(breadcrumb_stack.values())
        
        sections.append({
            "title": title,
            "level": level,
            "content": content,
            "breadcrumb": breadcrumb,
        })
    
    log.info(f"  {len(sections)} bölüm çıkarıldı")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Akıllı Chunking
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Markdown sözdizimini temizler, düz metin üretir."""
    # Tablo satırlarını birleştir
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'^[-\s|]+$', '', text, flags=re.MULTILINE)
    # Markdown kalın/italik
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    # Bağlantılar
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Başlık işaretleri
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Çoklu boşluklar
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def split_into_chunks(
    text: str,
    max_words: int,
    overlap_words: int
) -> List[str]:
    """
    Metni kelime bazlı, örtüşen chunk'lara böler.
    Cümle sınırlarına saygı gösterir (nokta/soru işareti/ünlem).
    """
    # Önce cümlelere böl
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_words = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        if current_word_count + sentence_word_count > max_words and current_words:
            # Mevcut chunk'ı kaydet
            chunks.append(" ".join(current_words))
            # Örtüşme için son N kelimeyi tut
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
    Her bölümü chunk'lara böler ve zengin metadata ekler.
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
            # Embedding için önek ekle (E5 modeli için gerekli)
            embed_text = f"passage: {section['breadcrumb']}\n{chunk_text}"
            
            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text,              # Ham metin (gösterim için)
                "embed_text": embed_text,         # Embedding için metin
                "section_title": section["title"],
                "breadcrumb": section["breadcrumb"],
                "section_level": section["level"],
                "chunk_index_in_section": i,
                "total_chunks_in_section": len(sub_chunks),
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
            })
            chunk_id += 1
    
    log.info(f"  Toplam {len(all_chunks)} chunk oluşturuldu")
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Embedding Üretimi
# ═══════════════════════════════════════════════════════════════════════════════

def generate_embeddings(chunks: List[Dict], model_name: str) -> np.ndarray:
    """
    Her chunk için embedding vektörü üretir.
    Büyük veri setlerinde batch processing kullanır.
    """
    log.info(f"Embedding modeli yükleniyor: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = [c["embed_text"] for c in chunks]
    total = len(texts)
    
    log.info(f"{total} chunk için embedding üretiliyor...")
    
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,   # Cosine similarity için normalize
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_embeddings.append(embeddings)
        log.info(f"  [{i + len(batch)}/{total}] chunk işlendi")
    
    result = np.vstack(all_embeddings).astype("float32")
    log.info(f"  Embedding boyutu: {result.shape}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FAISS Vektör Veritabanı
# ═══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Inner Product (cosine, çünkü vektörler normalize edildi) FAISS index'i oluşturur.
    Küçük/orta veri setleri için IndexFlatIP en güvenilir seçimdir.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"  FAISS index oluşturuldu: {index.ntotal} vektör, dim={dim}")
    return index


def save_artifacts(
    index: faiss.IndexFlatIP,
    chunks: List[Dict],
    config: Dict
) -> None:
    """Index ve metadata'yı diske kaydeder."""
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # FAISS index
    faiss.write_index(index, config["faiss_index_file"])
    log.info(f"  FAISS index kaydedildi: {config['faiss_index_file']}")
    
    # Chunks metadata (embed_text'i çıkar — disk alanı tasarrufu)
    chunks_for_save = [
        {k: v for k, v in c.items() if k != "embed_text"}
        for c in chunks
    ]
    with open(config["chunks_meta_file"], "w", encoding="utf-8") as f:
        json.dump(chunks_for_save, f, ensure_ascii=False, indent=2)
    log.info(f"  Chunk metadata kaydedildi: {config['chunks_meta_file']}")
    
    # Config snapshot (hangi model ve parametrelerle oluşturulduğunu kaydet)
    config_snapshot = {
        **config,
        "total_chunks": len(chunks),
        "embedding_dim": index.d,
        "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config["vectorstore_config_file"], "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)
    log.info(f"  Config snapshot kaydedildi: {config['vectorstore_config_file']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Ana Akış
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("BAL Chatbot — Vektör Veritabanı Oluşturma Başladı")
    log.info("=" * 60)
    
    t0 = time.time()
    
    # 1. Dokümanı oku
    markdown = load_markdown(CONFIG["dataset_path"])
    
    # 2. Bölümlere ayır
    log.info("Doküman bölümlere ayrılıyor...")
    sections = extract_sections(markdown)
    
    # 3. Chunk'lara böl
    log.info("Chunk'lar oluşturuluyor...")
    chunks = build_chunks(sections, CONFIG)
    
    # İstatistik
    word_counts = [c["word_count"] for c in chunks]
    log.info(
        f"  Chunk istatistikleri — "
        f"min: {min(word_counts)}, "
        f"max: {max(word_counts)}, "
        f"ort: {sum(word_counts) / len(word_counts):.0f} kelime"
    )
    
    # 4. Embedding üret
    embeddings = generate_embeddings(chunks, CONFIG["embedding_model"])
    
    # 5. FAISS index oluştur
    log.info("FAISS index oluşturuluyor...")
    index = build_faiss_index(embeddings)
    
    # 6. Kaydet
    log.info("Artifact'lar kaydediliyor...")
    save_artifacts(index, chunks, CONFIG)
    
    elapsed = time.time() - t0
    log.info(f"\n✅ Tamamlandı! Süre: {elapsed:.1f} saniye")
    log.info(f"   Toplam chunk: {len(chunks)}")
    log.info(f"   FAISS index: {CONFIG['faiss_index_file']}")
    log.info(f"   Chunk metadata: {CONFIG['chunks_meta_file']}")
    log.info("\nSonraki adım: python scripts/02_chatbot.py")


if __name__ == "__main__":
    main()
