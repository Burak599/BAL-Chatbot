---
title: BAL Asistan
emoji: 🏫
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# BAL Chatbot — RAG-Powered AI Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-00C853?logo=databricks&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-LLM_Inference-F97316?logo=groq&logoColor=white)](https://groq.com)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Spaces-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Bornova Anadolu Lisesi için Retrieval-Augmented Generation (RAG) tabanlı yapay zeka asistanı.**  
> Flask + FAISS + yerel embedding + Groq LLM. Yanıtlar SSE ile akış halinde gelir.  
> Ziyaretçiler FingerprintJS ile anonim takip edilir; kota ve sohbet logları veritabanında tutulur.

**Canlı:** [brk9999/bal-chatbot](https://huggingface.co/spaces/brk9999/bal-chatbot) · `https://brk9999-bal-chatbot.hf.space`

---

## Özellikler

- **RAG pipeline** — Okul bilgi tabanından FAISS ile semantik arama (top-5, eşik ≥ 0.35)
- **Yerel embedding** — `intfloat/multilingual-e5-small` (384 boyut); harici embedding API’si gerekmez
- **SSE streaming** — Groq üzerinden gerçek zamanlı token akışı
- **Model / key failover** — 4 modellik zincir; birden fazla Groq API anahtarı desteği
- **Anonim ziyaretçi kimliği** — FingerprintJS; kayıt zorunlu değil
- **Kota sistemi** — Rol bazlı günlük ve dakikalık limitler
- **Oturum hafızası** — Oturum başına son 6 tur (12 mesaj)
- **Geri bildirim** — Like / dislike ve serbest metin; `chat_logs` tablosuna yazılır
- **Yoğunluk uyarısı** — Eşzamanlı istek eşiği aşılınca istemciye bildirim
- **SPA arayüz** — Vanilla HTML/CSS/JS, responsive tasarım

---

## Mimari

```
Kullanıcı ──▶ FingerprintJS ──▶ Flask API ──▶ Kota kontrolü
                                              │
                                    FAISS arama (top-5)
                                    e5-small yerel embedding
                                              │
                                    Bağlam + soru (prompt)
                                              │
                              ┌───────────────┴───────────────┐
                              │  Groq LLM Gateway             │
                              │  model zinciri + API key pool │
                              │  curl_cffi (hosted ortamlar)  │
                              └───────────────┬───────────────┘
                                              │
                                    SSE stream ──▶ İstemci
                                              │
                              Oturum geçmişi (bellek) + loglar (DB)
```

### Teknoloji

| Katman | Teknoloji |
|--------|-----------|
| **Backend** | Python 3.11+, Flask 3.0 |
| **Vektör arama** | FAISS (`IndexFlatIP`, normalize edilmiş cosine) |
| **Embedding** | `sentence-transformers` → `intfloat/multilingual-e5-small` (384-dim) |
| **LLM** | Groq API (`llama-3.3-70b-versatile` + yedek model zinciri) |
| **Veritabanı** | Neon PostgreSQL (önerilen) veya SQLite (yedek) |
| **Frontend** | Vanilla JS + FingerprintJS |
| **Deploy** | **Hugging Face Spaces** (Docker) |

---

## Proje yapısı

```
ChatbotBAL/
├── web/
│   ├── app.py              # Flask API — RAG, LLM, auth, kota, route'lar
│   ├── index.html          # SPA frontend
│   ├── wsgi.py             # Alternatif WSGI giriş noktası
│   ├── vendor/             # FingerprintJS (vendored)
│   └── BAL_Logo.png
├── scripts/
│   ├── 01_build_vectorstore.py   # Markdown → chunk → FAISS index
│   ├── 02_chatbot.py             # Terminal CLI chatbot
│   ├── 03_eval_retrieval.py      # Retrieval kalite değerlendirmesi
│   ├── locustfile.py             # Locust yük testi
│   └── test_concurrent.py        # 20 eşzamanlı bot testi (HF Space)
├── data/
│   ├── bal_faiss.index           # FAISS vektör indeksi
│   ├── bal_chunks.json           # Chunk metadata (~142 parça)
│   ├── vectorstore_config.json   # Index build özeti
│   └── app.db                    # SQLite (DATABASE_URL yoksa)
├── Dataset/
│   └── RAG_Dataset_BAL.md        # Kaynak bilgi tabanı
├── docs/
│   └── neon_postgresql.md        # Neon PostgreSQL kurulumu
├── .github/workflows/deploy.yml    # main → HF Space otomatik yükleme
├── .env.example                    # Ortam değişkenleri şablonu
├── Dockerfile                      # HF Space container (python web/app.py)
├── requirements.txt
└── package.json                    # FingerprintJS bağımlılığı
```

---

## Nasıl çalışır?

1. `Dataset/RAG_Dataset_BAL.md` markdown dosyası bölümlere ayrılır ve chunk’lara bölünür.
2. `scripts/01_build_vectorstore.py` her chunk için yerel embedding üretir ve `data/bal_faiss.index` oluşturur.
3. Kullanıcı sorusu geldiğinde `web/app.py` aynı modelle sorguyu embed eder, FAISS’ten en alakalı parçaları bulur.
4. Bağlam + sistem prompt’u + son 6 tur geçmiş Groq’a gönderilir.
5. Yanıt SSE ile akar; başarılı cevaplar `chat_logs` tablosuna yazılır.

**Önemli:** Embedding modeli index build sırasında ve runtime’da aynı olmalı (`intfloat/multilingual-e5-small`).

---

## Ortam değişkenleri

Şablon dosya: [`.env.example`](.env.example)

Yerel geliştirme:
```bash
cp .env.example .env
# .env dosyasını düzenle — en az GROQ_API_KEY doldur
```

HF Space → **Settings → Repository secrets** veya Space **Variables** üzerinden tanımlanır.

| Değişken | Zorunlu | Açıklama |
|----------|---------|----------|
| `GROQ_API_KEY` | Evet* | Birincil Groq API anahtarı |
| `GROQ_API_KEYS` | Hayır | Virgülle ayrılmış birden fazla anahtar |
| `GROQ_API_KEY_1` … `GROQ_API_KEY_5` | Hayır | Ek anahtar slotları |
| `GROQ_MODEL_CHAIN` | Hayır | Virgülle ayrılmış model listesi (varsayılan 4 model) |
| `DATABASE_URL` | Önerilir | Neon PostgreSQL connection string |
| `FLASK_SECRET_KEY` | Önerilir | Oturum çerezi imzası |
| `ADMIN_EMAILS` | Hayır | Admin rolü için e-posta listesi (virgülle) |
| `FORCE_HTTPS` | Hayır | `true` ise HTTP → HTTPS yönlendirmesi |
| `LOCAL_HTTPS` | Hayır | Yerel geliştirmede HTTPS (`adhoc` sertifika) |
| `PORT` | Hayır | Sunucu portu (yerelde varsayılan `7860`) |
| `GOOGLE_CLIENT_ID` | Hayır | Google auth (şu an endpoint’ler devre dışı) |
| `HF_TOKEN` | Deploy* | GitHub Actions → HF Space yükleme token’ı |

\* En az bir Groq anahtarı gerekir (`GROQ_API_KEY`, `GROQ_API_KEYS` veya `GROQ_API_KEY_1` vb.).

\* `HF_TOKEN` yalnızca GitHub Actions deploy için gerekir; uygulama runtime’ında kullanılmaz.

`DATABASE_URL` yoksa veya bağlantı başarısızsa uygulama `data/app.db` SQLite’a düşer.

Tüm alanlar ve yorumlar için bkz. [`.env.example`](.env.example).

Detaylı PostgreSQL kurulumu: [`docs/neon_postgresql.md`](docs/neon_postgresql.md)

---

## Geliştirme

### Gereksinimler

- Python 3.11+
- Groq API anahtarı — [console.groq.com](https://console.groq.com)
- (Önerilen) Neon PostgreSQL — kalıcı kullanıcı / kota / log için

> Embedding için **Gemini veya başka bir API gerekmez**; model yerel çalışır.

### Kurulum

```bash
git clone https://github.com/Burak599/BAL-Chatbot.git
cd BAL-Chatbot

pip install -r requirements.txt

# Vektör veritabanını oluştur (dataset değiştiyse)
python scripts/01_build_vectorstore.py

# Ortam değişkenlerini ayarla
cp .env.example .env
# .env içinde en az GROQ_API_KEY doldur

python web/app.py
```

Varsayılan port: `7860` (HF Space ile uyumlu).

---

## Hugging Face Spaces deploy

Bu proje **yalnızca Hugging Face Spaces** üzerinde canlıya alınır.

| Öğe | Değer |
|-----|-------|
| Space | [brk9999/bal-chatbot](https://huggingface.co/spaces/brk9999/bal-chatbot) |
| SDK | Docker (`Dockerfile`) |
| Başlatma | `python web/app.py` |
| Otomatik deploy | `main` branch’e push → GitHub Actions |

### Deploy akışı

1. Değişiklikleri `main` branch’ine push et.
2. `.github/workflows/deploy.yml` tetiklenir.
3. Workflow, repoyu `HF_TOKEN` ile `brk9999/bal-chatbot` Space’ine yükler.
4. Space yeniden build eder ve container’ı başlatır.

### GitHub secret

Repository → **Settings → Secrets → Actions**:

```
HF_TOKEN = Hugging Face write token
```

### Space ayarları

- **Hardware:** 2 vCPU / 16 GB RAM (embedding modeli için startup’ta yüklenir)
- **Secrets / Variables:** `GROQ_API_KEY`, `DATABASE_URL`, `FLASK_SECRET_KEY` vb.

### Rollback

Deploy sonrası sorun olursa GitHub’daki önceki çalışan commit’e `git revert` ile dönüp tekrar `main`’e push edebilirsin; workflow eski kodu Space’e yükler.

---

## API

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/` | Frontend (`index.html`) |
| `GET` | `/api/health` | Sağlık kontrolü (vectorstore, DB, Groq) |
| `GET` | `/api/auth/status` | Kimlik, rol ve kota bilgisi |
| `POST` | `/api/chat` | SSE streaming sohbet |
| `POST` | `/api/chat/feedback` | Yanıt geri bildirimi |

`POST /api/auth/guest`, `/register`, `/login`, `/google`, `/logout` endpoint’leri şu an **devre dışı** (404).

### `POST /api/chat`

**İstek:**
```json
{
  "message": "LGS taban puanı nedir?",
  "session_id": "session_abc123"
}
```

**Header (önerilir):**
```
X-Client-Fingerprint: <fingerprintjs-id>
Content-Type: application/json
```

**Yanıt (SSE):**
```
data: {"token": "2025"}
data: {"token": " LGS"}
data: {"done": true, "sources": [...], "question_index": 5, "near_limit": false}
```

---

## Kota limitleri

| Rol | Günlük | Dakikalık |
|-----|--------|-----------|
| Ziyaretçi (`visitor`) | 40 | 5 |
| Kullanıcı (`user`) | 50 | 8 |
| Admin (`admin`) | 500 | 20 |

Ziyaretçiler fingerprint ile otomatik oluşturulur (`provider = fingerprint`, `role = visitor`).

---

## Test araçları

```bash
# Retrieval kalite raporu → logs/eval_report.txt
python scripts/03_eval_retrieval.py

# 20 eşzamanlı bot (HF Space URL’si scripts/test_concurrent.py içinde)
python scripts/test_concurrent.py

# Locust (ayrı kurulum: pip install locust gevent)
locust -f scripts/locustfile.py --host https://brk9999-bal-chatbot.hf.space --users 20 --spawn-rate 20 --headless
```

---

## Lisans

MIT License — ayrıntılar için [`LICENSE`](LICENSE).

---

<p align="center">
  <sub>Bornova Anadolu Lisesi · Burak599</sub>
</p>
