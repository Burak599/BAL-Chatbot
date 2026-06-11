"""
═══════════════════════════════════════════════════════════════════════════════
 BAL Chatbot — 20 Eşzamanlı Kullanıcı Yük Testi (Gecikme Debug)

 Amaç:
   20 bot aynı anda 1'er soru gönderir.
   Her bot için cevap süresi ve TTFT ölçülür.
   Tüm botlar tamamlandığında özet tablo basılır.

 Kullanım:
   locust -f locustfile.py --host https://chatbot-bal.onrender.com --users 20 --spawn-rate 20 --headless --run-time 120s

   Lokal test (localhost:5000):
   locust -f locustfile.py --host http://localhost:5000 --users 20 --spawn-rate 20 --headless --run-time 120s
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import random
import string
import time
import uuid
import threading
import logging
import sys

import gevent
from gevent.event import Event

from locust import HttpUser, task, constant
from locust.exception import StopUser

# ── Loglama (hem dosyaya hem ekrana) ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("load_test_results.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("bal_load_test")

# ── Konfigürasyon ────────────────────────────────────────────────────────────
NUM_CLIENTS = 20               # Toplam bot sayısı
REQUEST_TIMEOUT = 180          # Saniye cinsinden maksimum bekleme

# ── Senkronizasyon: Tüm botlar spawn olana kadar bekleme ─────────────────────
_ready_event = Event()
_ready_counter = [0]           # Mutable list (closure için)

# ── Sonuç toplama (thread-safe) ──────────────────────────────────────────────
results = []
results_lock = threading.Lock()


# ── Soru havuzu ──────────────────────────────────────────────────────────────
QUESTIONS = [
    "LGS taban puanı nedir?",
    "Okula nasıl kayıt yapılır?",
    "BALEV bursu hakkında bilgi ver",
    "Okula nasıl giderim?",
    "YKS başarıları nasıl?",
    "Ayran Günü nedir?",
    "Okuldaki kulüpler nelerdir?",
    "Pansiyon imkânı var mı?",
    "Yabancı dil eğitimi nasıl?",
    "Mezun öğrenciler ne yapıyor?",
    "Okulun başarıları nelerdir?",
    "BAL nedir?",
    "Okulda hangi etkinlikler var?",
    "BALMED nedir?",
    "Okul saat kaçta başlıyor?",
    "Hazırlık sınıfı var mı?",
    "Okulun tarihi nedir?",
    "Sportif başarılar neler?",
    "Okulda hangi diller öğretiliyor?",
    "BALEV nedir?",
]


def print_summary_table():
    """Tüm botlar tamamlandıktan sonra özet tabloyu basar."""
    if not results:
        log.info("\n❌ Hiçbir sonuç kaydedilmedi.")
        return

    # ── Log dosyasına detaylı tablo ──────────────────────────────────────────
    log.info("\n" + "═" * 90)
    log.info("📊 20 EŞZAMANLI KULLANICI — DETAYLI SONUÇ TABLOSU")
    log.info("═" * 90)
    log.info(f"{'Bot #':<6} {'TTFT (ms)':<12} {'Toplam (ms)':<14} {'Token':<7} {'Soru':<50}")
    log.info("─" * 90)

    total_ttft = []
    total_duration = []
    total_tokens = []

    for r in sorted(results, key=lambda x: x["bot_id"]):
        ttft_ms = round(r["ttft"] * 1000) if r["ttft"] is not None else "N/A"
        dur_ms = round(r["duration"] * 1000)
        token_str = str(r["tokens"])

        if r["ttft"] is not None:
            total_ttft.append(r["ttft"])
        total_duration.append(r["duration"])
        total_tokens.append(r["tokens"])

        status = "✅" if not r["error"] else "❌"
        q_short = (r["question"][:47] + "...") if len(r["question"]) > 48 else r["question"]
        log.info(f"{status} {r['bot_id']:<4} {ttft_ms!s:<12} {dur_ms!s:<14} {token_str:<7} {q_short:<50}")

    log.info("─" * 90)

    # ── İstatistikler ────────────────────────────────────────────────────────
    if total_ttft:
        avg_ttft = sum(total_ttft) / len(total_ttft)
        min_ttft = min(total_ttft)
        max_ttft = max(total_ttft)
    else:
        avg_ttft = min_ttft = max_ttft = 0

    avg_dur = sum(total_duration) / len(total_duration)
    min_dur = min(total_duration)
    max_dur = max(total_duration)
    avg_tok = sum(total_tokens) / len(total_tokens) if total_tokens else 0
    success_count = sum(1 for r in results if not r["error"])
    fail_count = sum(1 for r in results if r["error"])

    log.info("─" * 90)
    log.info("📈 ÖZET İSTATİSTİKLER:")
    log.info(f"   Başarılı / Toplam   : {success_count} / {len(results)}")
    log.info(f"   Başarısız            : {fail_count}")
    log.info(f"")
    log.info(f"   📌 TTFT (İlk Token Süresi):")
    log.info(f"      Ortalama  : {avg_ttft*1000:.0f} ms")
    log.info(f"      Minimum   : {min_ttft*1000:.0f} ms")
    log.info(f"      Maksimum  : {max_ttft*1000:.0f} ms")
    log.info(f"")
    log.info(f"   📌 Toplam Yanıt Süresi:")
    log.info(f"      Ortalama  : {avg_dur*1000:.0f} ms")
    log.info(f"      Minimum   : {min_dur*1000:.0f} ms")
    log.info(f"      Maksimum  : {max_dur*1000:.0f} ms")
    log.info(f"")
    log.info(f"   📌 Token Sayısı:")
    log.info(f"      Ortalama  : {avg_tok:.1f}")
    avg_tps = avg_tok / avg_dur if avg_dur > 0 else 0
    log.info(f"   📌 Ortalama Token/s : {avg_tps:.1f}")
    log.info("═" * 90)


class BALChatUser(HttpUser):
    """Her bir bot (sanal kullanıcı)."""

    # Botlar arasında bekleme YOK — senkronizasyon on_start ile sağlanır
    wait_time = constant(0)

    def on_start(self):
        """
        Tüm botlar spawn olana kadar bekler, sonra hepsi aynı anda serbest kalır.
        20. bot spawn olduğunda _ready_event set edilir.
        """
        # Benzersiz fingerprint (her bot için ayrı kimlik)
        self.fingerprint = "".join(random.choices(string.ascii_lowercase + string.digits, k=32))
        self.client.headers.update({"X-Client-Fingerprint": self.fingerprint})

        # Global sayaç (gevent-safe: mutable list)
        _ready_counter[0] += 1
        current_count = _ready_counter[0]

        self.bot_id = current_count  # Bot numarası (1-20)

        log.info(f"🔄 Bot #{self.bot_id} spawn oldu ({current_count}/{NUM_CLIENTS})")

        # Tüm botlar hazır mı?
        if current_count >= NUM_CLIENTS:
            log.info(f"🚦 Tüm {NUM_CLIENTS} bot hazır! Aynı anda fırlatılıyor...")
            _ready_event.set()
        else:
            # Diğer botları bekle
            _ready_event.wait(timeout=60)

        self.done = False

    @task
    def send_one_question(self):
        if self.done:
            raise StopUser()

        self.done = True
        question = random.choice(QUESTIONS)

        # ⏱ 1. AN: Sorunun gönderildiği tam an
        start_time = time.time()

        first_token_at = None
        end_time = None
        tokens = 0
        error = None

        try:
            with self.client.post(
                "/api/chat",
                json={
                    "message": question,
                    "session_id": f"bot_{self.bot_id}_{uuid.uuid4().hex[:8]}",
                },
                headers={"Accept": "text/event-stream"},
                stream=True,
                catch_response=True,
                timeout=REQUEST_TIMEOUT,
                name="/api/chat",
            ) as resp:
                if resp.status_code != 200:
                    error = f"HTTP {resp.status_code}"
                    resp.failure(error)
                else:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        line = line.decode() if isinstance(line, bytes) else line
                        if not line.startswith("data: "):
                            continue

                        try:
                            data = json.loads(line[6:])
                        except Exception:
                            continue

                        if "token" in data:
                            if first_token_at is None:
                                first_token_at = time.time()
                            tokens += 1

                        if "error" in data:
                            error = data["error"]

                        if data.get("done"):
                            end_time = time.time()
                            break

                    if end_time is None:
                        end_time = time.time()

                    resp.success()

        except Exception as e:
            error = str(e)
            end_time = time.time()

        if end_time is None:
            end_time = time.time()

        # ── Süre hesaplamaları ───────────────────────────────────────────────
        duration = end_time - start_time
        ttft = first_token_at - start_time if first_token_at else None

        # ── Sonucu kaydet ────────────────────────────────────────────────────
        with results_lock:
            results.append({
                "bot_id": self.bot_id,
                "ttft": ttft,
                "duration": duration,
                "tokens": tokens,
                "question": question,
                "error": error,
            })

        # ── Anlık log ────────────────────────────────────────────────────────
        status_icon = "✅" if not error else "❌"
        ttft_str = f"{ttft*1000:.0f}ms" if ttft is not None else "N/A"
        log.info(
            f"{status_icon} Bot #{self.bot_id:>2} | "
            f"TTFT={ttft_str:>7} | "
            f"Toplam={duration*1000:.0f}ms | "
            f"Token={tokens:>3} | "
            f"Soru='{question[:30]}'"
        )

        # ── En son biten bot özet tabloyu bassın ────────────────────────────
        with results_lock:
            if len(results) >= NUM_CLIENTS:
                log.info("\n" + "🔥" * 20 + " TÜM BOTLAR TAMAMLANDI " + "🔥" * 20)
                if len(results) == NUM_CLIENTS:
                    print_summary_table()

        raise StopUser()