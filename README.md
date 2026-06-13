# 🏫 BAL Chatbot — RAG-Powered AI Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-00C853?logo=databricks&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-LLM_Inference-F97316?logo=groq&logoColor=white)](https://groq.com)
[![Gemini](https://img.shields.io/badge/Gemini-Embedding-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Render](https://img.shields.io/badge/Render-Deploy-46E3B7?logo=render&logoColor=white)](https://render.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A Retrieval-Augmented Generation (RAG) chatbot for Bornova Anadolu Lisesi.**  
> Built with Flask, FAISS, Groq LLMs, and Gemini embeddings.  
> Streams responses via SSE. Supports anonymous fingerprint-based auth with per-user rate limits.

---

## ✨ Features

- **🔍 RAG Pipeline** — Semantic search over 135 document chunks using FAISS (cosine similarity)
- **⚡ Streaming Responses** — Server-Sent Events for real-time token delivery
- **🧠 Multi-Model Failover** — 4-model chain with 6 API keys for high availability
- **🛡️ Cloudflare Bypass** — `curl_cffi` with Chrome impersonation for Render deployments
- **👤 Anonymous Auth** — FingerprintJS-based visitor tracking, no sign-up required
- **📊 Rate Limiting** — Per-role quotas (visitor/user/admin) with daily & minute windows
- **🔄 Conversation Memory** — Last 6 turns preserved per session
- **📝 User Feedback** — Like/dislike + free-text feedback persisted to database
- **🖥️ Modern SPA Frontend** — Vanilla JS, responsive design, glassmorphism UI
- **🧪 Load Testing** — Locust script for 20-concurrent-user stress tests
- **🐳 Dockerized** — Single-command deployment

---

## 🏗 Architecture

```
User ──▶ FingerprintJS ──▶ Flask API ──▶ Rate Limit Check
                                              │
                                         FAISS Search
                                         (top-5 chunks)
                                              │
                                      Context Builder
                                      (threshold ≥ 0.35)
                                              │
                                     ┌────────┴────────┐
                                     │  LLM Gateway    │
                                     │  (Groq)         │
                                     │  Model Chain    │
                                     │  API Key Pool   │
                                     └────────┬────────┘
                                              │
                                      SSE Stream ──▶ Client
                                              │
                                      History & Logs
                                      (PostgreSQL/SQLite)
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11+, Flask 3.0, Gunicorn |
| **Vector DB** | FAISS (IndexFlatIP, 3072-dim embeddings) |
| **Embeddings** | Google Gemini (`models/gemini-embedding-001`) |
| **LLM** | Groq API (`llama-3.3-70b-versatile` + fallback chain) |
| **Database** | Neon PostgreSQL / SQLite (dev) |
| **Frontend** | Vanilla HTML/CSS/JS, FingerprintJS |
| **Deploy** | Docker, Render, Hugging Face Spaces |

---

## 📁 Project Structure

```
ChatbotBAL/
├── web/
│   ├── app.py              # Flask API (1,416 LOC)
│   ├── index.html          # SPA frontend (1,891 LOC)
│   ├── wsgi.py             # Gunicorn entry point
│   └── BAL_Logo.png        # Brand asset
├── scripts/
│   ├── 01_build_vectorstore.py   # Build FAISS index from markdown
│   ├── 02_chatbot.py             # CLI chatbot (terminal)
│   ├── 03_eval_retrieval.py      # Retrieval quality evaluation
│   └── locustfile.py             # Load testing (20 concurrent users)
├── data/
│   ├── bal_faiss.index           # FAISS vector index
│   ├── bal_chunks.json           # Chunk metadata (135 chunks)
│   └── vectorstore_config.json   # Build configuration snapshot
├── Dataset/
│   └── RAG_Dataset_BAL.md        # Source knowledge base (1,647 lines)
├── docs/
│   └── neon_postgresql.md        # Database setup guide
├── Dockerfile                    # Python 3.11-slim container
├── render.yaml                   # Render deployment config
├── .github/workflows/deploy.yml  # Hugging Face auto-deploy
├── requirements.txt              # Python dependencies
├── netlify.toml                  # Static file hosting config
└── package.json                  # FingerprintJS dependency
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.11+
- Groq API key(s) — [groq.com](https://console.groq.com)
- Gemini API key(s) — [aistudio.google.com](https://aistudio.google.com)
- (Optional) Neon PostgreSQL database

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Burak599/BAL-Chatbot.git
cd BAL-Chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the vector store
python scripts/01_build_vectorstore.py

# 4. Configure environment variables
cp .env.example .env
# Edit .env — set GROQ_API_KEY, GEMINI_API_KEY, etc.

# 5. Run the web server
python web/app.py
```

### Docker

```bash
docker build -t bal-chatbot .
docker run -p 7860:7860 --env-file .env bal-chatbot
```

---

## 🚀 API Reference

### `POST /api/chat`
Stream a chat response.

**Request:**
```json
{
  "message": "LGS taban puanı nedir?",
  "session_id": "session_abc123"
}
```

**Response (SSE):**
```
data: {"token": "2025"}
data: {"token": " LGS"}
data: {"token": " puanı..."}
data: {"done": true, "sources": [...], "question_index": 5}
```

### `GET /api/health`
System health check.

### `GET /api/auth/status`
Current user identity, role, and quota info.

### `POST /api/chat/feedback`
Submit feedback on a response.

---

## 🧪 Testing

```bash
# Retrieval quality evaluation
python scripts/03_eval_retrieval.py

# Load testing (local)
locust -f scripts/locustfile.py --host http://localhost:5000 --users 20 --spawn-rate 20 --headless

# Load testing (production)
locust -f scripts/locustfile.py --host https://your-app.onrender.com --users 20 --spawn-rate 20 --headless
```

---

## 🌍 Deployment

| Platform | Config | Notes |
|----------|--------|-------|
| **Render** | `render.yaml` | Free tier, Gunicorn with 8 threads |
| **Hugging Face** | `.github/workflows/deploy.yml` | Auto-deploy on push to `main` |
| **Netlify** | `netlify.toml` | Static file serving (frontend only) |
| **Docker** | `Dockerfile` | Custom container, any cloud |

---

## 🔐 Rate Limits

| Role | Daily Limit | Minute Limit |
|------|-------------|--------------|
| 🧑 Visitor | 40 | 5 |
| 👤 User | 50 | 8 |
| 🔧 Admin | 500 | 20 |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ by Burak599</sub>
</p>