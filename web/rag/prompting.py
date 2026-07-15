"""
Prompting utilities for RAG context formatting.

Usage: Import functions directly, e.g.:
    from rag.prompting import format_context, build_augmented_user_message, build_sources_payload
"""

from typing import List, Dict


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
