"""
=============================================================
 BAL Chatbot — Adım 3: Retrieval Kalite Testi
 Kullanım: python scripts/03_eval_retrieval.py
=============================================================
Bu script RAG sisteminin retrieval kalitesini test eder:
  - Önceden hazırlanmış test sorularını çalıştırır
  - Her soru için retrieve edilen chunk'ları değerlendirir
  - Raporu logs/eval_report.txt dosyasına yazar
=============================================================
"""

import json
import time
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Test Soruları ─────────────────────────────────────────────────────────────
# Her soru için beklenen anahtar kelimeler (değerlendirme için)
TEST_QUESTIONS = [
    {
        "question": "BAL'ın kuruluş tarihi nedir?",
        "expected_keywords": ["1953", "Ege Koleji", "Giraud"],
    },
    {
        "question": "Almanca bölümünün LGS taban puanı kaçtır?",
        "expected_keywords": ["484", "Almanca", "taban"],
    },
    {
        "question": "Ayran Günü nedir?",
        "expected_keywords": ["Mayıs", "şenlik", "geleneksel", "müzik"],
    },
    {
        "question": "BALEV bursuna nasıl başvurabilirim?",
        "expected_keywords": ["balev.org.tr", "burs", "başvuru"],
    },
    {
        "question": "Okula metro ile nasıl gidebilirim?",
        "expected_keywords": ["Bornova Metro", "otobüs", "267", "268"],
    },
    {
        "question": "Ultimate Frizbi takımı var mı?",
        "expected_keywords": ["Ultimate Frizbi", "tek lise", "BALspor"],
    },
    {
        "question": "Hazırlık sınıfında ne öğretilir?",
        "expected_keywords": ["yabancı dil", "yoğunlaştırılmış", "hazırlık"],
    },
    {
        "question": "DSD diploması nedir?",
        "expected_keywords": ["Deutsches Sprachdiplom", "Almanca", "diploma"],
    },
    {
        "question": "Okulun vizyon cümlesi nedir?",
        "expected_keywords": ["Geleceğin Aydınlık Sesi", "vizyon"],
    },
    {
        "question": "Pansiyon ücreti ne kadar?",
        "expected_keywords": ["pansiyon", "güncel veri"],
    },
]


def load_artifacts(index_path: str, chunks_path: str, model_name: str):
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    model = SentenceTransformer(model_name)
    return index, chunks, model


def retrieve(query: str, index, chunks, model, top_k: int = 5):
    query_text = f"query: {query}"
    embedding = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def evaluate_retrieval(results, expected_keywords: list) -> dict:
    """Retrieve edilen chunk'larda beklenen kelimelerin bulunup bulunmadığını kontrol eder."""
    combined_text = " ".join(r.get("text", "").lower() for r in results)
    found = [kw for kw in expected_keywords if kw.lower() in combined_text]
    recall = len(found) / len(expected_keywords) if expected_keywords else 1.0
    return {
        "found_keywords": found,
        "missing_keywords": [kw for kw in expected_keywords if kw not in found],
        "recall": recall,
        "top_score": results[0]["score"] if results else 0.0,
        "avg_score": sum(r["score"] for r in results) / len(results) if results else 0.0,
    }


def run_evaluation():
    # Yükle
    print("Artifact'lar yükleniyor...")
    try:
        index, chunks, model = load_artifacts(
            "data/bal_faiss.index",
            "data/bal_chunks.json",
            "intfloat/multilingual-e5-large",
        )
    except FileNotFoundError:
        print("❌ Vektör veritabanı bulunamadı. Önce 01_build_vectorstore.py çalıştırın.")
        return
    
    print(f"  ✓ {index.ntotal} chunk yüklendi\n")
    
    report_lines = [
        "BAL Chatbot — Retrieval Kalite Raporu",
        "=" * 60,
        f"Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Toplam test sorusu: {len(TEST_QUESTIONS)}",
        "",
    ]
    
    recalls = []
    top_scores = []
    
    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        expected = test["expected_keywords"]
        
        results = retrieve(question, index, chunks, model, top_k=5)
        eval_result = evaluate_retrieval(results, expected)
        
        recalls.append(eval_result["recall"])
        top_scores.append(eval_result["top_score"])
        
        # Terminal çıktı
        status = "✅" if eval_result["recall"] >= 0.7 else "⚠" if eval_result["recall"] >= 0.4 else "❌"
        print(f"{status} [{i:02d}] {question}")
        print(f"       Recall: {eval_result['recall']:.0%} | Top Score: {eval_result['top_score']:.3f}")
        if eval_result["missing_keywords"]:
            print(f"       Eksik: {eval_result['missing_keywords']}")
        print()
        
        # Rapor
        report_lines += [
            f"── Soru {i}: {question}",
            f"   Recall      : {eval_result['recall']:.0%}",
            f"   Top Score   : {eval_result['top_score']:.3f}",
            f"   Avg Score   : {eval_result['avg_score']:.3f}",
            f"   Bulunan kw  : {eval_result['found_keywords']}",
            f"   Eksik kw    : {eval_result['missing_keywords']}",
            "",
            "   Chunk başlıkları:",
        ]
        for r in results[:3]:
            report_lines.append(
                f"     [{r['score']:.3f}] {r.get('breadcrumb', '')} — {r.get('text', '')[:80]}..."
            )
        report_lines.append("")
    
    # Özet
    avg_recall = sum(recalls) / len(recalls)
    avg_top = sum(top_scores) / len(top_scores)
    
    summary = [
        "=" * 60,
        "ÖZET",
        f"  Ortalama Recall  : {avg_recall:.0%}",
        f"  Ortalama Top Skor: {avg_top:.3f}",
        f"  Başarılı (≥70%)  : {sum(1 for r in recalls if r >= 0.7)}/{len(recalls)}",
    ]
    
    print("\n".join(summary))
    report_lines += [""] + summary
    
    # Kaydet
    Path("logs").mkdir(exist_ok=True)
    report_path = "logs/eval_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n📄 Rapor kaydedildi: {report_path}")


if __name__ == "__main__":
    run_evaluation()