"""
═══════════════════════════════════════════════════════════════════════════════
 BAL Chatbot — 20 Concurrent User Load Test (Latency Debug)

 Purpose:
   20 bots each send 1 question simultaneously.
   Response time and TTFT (Time to First Token) are measured per bot.
   A summary table is printed when all bots finish.

 Usage:
   locust -f locustfile.py --host https://chatbot-bal.onrender.com --users 20 --spawn-rate 20 --headless --run-time 120s

   Local test (localhost:5000):
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

# ── Logging (file + console) ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("load_test_results.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("bal_load_test")

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_CLIENTS = 20               # Total number of simulated bots
REQUEST_TIMEOUT = 180          # Max wait time in seconds

# ── Synchronisation: wait until all bots have spawned ────────────────────────
_ready_event = Event()
_ready_counter = [0]           # Mutable list (for closure compatibility)

# ── Result collection (thread-safe) ───────────────────────────────────────────
results = []
results_lock = threading.Lock()


# ── Question pool ─────────────────────────────────────────────────────────────
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
    """Prints a summary table after all bots have completed."""
    if not results:
        log.info("\n❌ No results recorded.")
        return

    # ── Detailed table to log file ────────────────────────────────────────────
    log.info("\n" + "═" * 90)
    log.info("📊 20 CONCURRENT USERS — DETAILED RESULTS TABLE")
    log.info("═" * 90)
    log.info(f"{'Bot #':<6} {'TTFT (ms)':<12} {'Total (ms)':<14} {'Tokens':<7} {'Question':<50}")
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

    # ── Statistics ────────────────────────────────────────────────────────────
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
    log.info("📈 SUMMARY STATISTICS:")
    log.info(f"   Success / Total   : {success_count} / {len(results)}")
    log.info(f"   Failed            : {fail_count}")
    log.info(f"")
    log.info(f"   📌 TTFT (Time to First Token):")
    log.info(f"      Average  : {avg_ttft*1000:.0f} ms")
    log.info(f"      Minimum  : {min_ttft*1000:.0f} ms")
    log.info(f"      Maximum  : {max_ttft*1000:.0f} ms")
    log.info(f"")
    log.info(f"   📌 Total Response Time:")
    log.info(f"      Average  : {avg_dur*1000:.0f} ms")
    log.info(f"      Minimum  : {min_dur*1000:.0f} ms")
    log.info(f"      Maximum  : {max_dur*1000:.0f} ms")
    log.info(f"")
    log.info(f"   📌 Token Count:")
    log.info(f"      Average  : {avg_tok:.1f}")
    avg_tps = avg_tok / avg_dur if avg_dur > 0 else 0
    log.info(f"   📌 Average Tokens/s : {avg_tps:.1f}")
    log.info("═" * 90)


class BALChatUser(HttpUser):
    """Each simulated bot (virtual user)."""

    # No wait between bots — synchronisation is handled via on_start
    wait_time = constant(0)

    def on_start(self):
        """
        Waits until all bots have spawned, then they all start simultaneously.
        The 20th bot sets _ready_event to release all others.
        """
        # Unique fingerprint (separate identity per bot)
        self.fingerprint = "".join(random.choices(string.ascii_lowercase + string.digits, k=32))
        self.client.headers.update({"X-Client-Fingerprint": self.fingerprint})

        # Global counter (gevent-safe via mutable list)
        _ready_counter[0] += 1
        current_count = _ready_counter[0]

        self.bot_id = current_count  # Bot number (1-20)

        log.info(f"🔄 Bot #{self.bot_id} spawned ({current_count}/{NUM_CLIENTS})")

        # Are all bots ready?
        if current_count >= NUM_CLIENTS:
            log.info(f"🚦 All {NUM_CLIENTS} bots ready! Launching simultaneously...")
            _ready_event.set()
        else:
            # Wait for other bots
            _ready_event.wait(timeout=60)

        self.done = False

    @task
    def send_one_question(self):
        if self.done:
            raise StopUser()

        self.done = True
        question = random.choice(QUESTIONS)

        # ⏱ Mark exact send time
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

        # ── Duration calculations ─────────────────────────────────────────────
        duration = end_time - start_time
        ttft = first_token_at - start_time if first_token_at else None

        # ── Save result ───────────────────────────────────────────────────────
        with results_lock:
            results.append({
                "bot_id": self.bot_id,
                "ttft": ttft,
                "duration": duration,
                "tokens": tokens,
                "question": question,
                "error": error,
            })

        # ── Live log ──────────────────────────────────────────────────────────
        status_icon = "✅" if not error else "❌"
        ttft_str = f"{ttft*1000:.0f}ms" if ttft is not None else "N/A"
        log.info(
            f"{status_icon} Bot #{self.bot_id:>2} | "
            f"TTFT={ttft_str:>7} | "
            f"Total={duration*1000:.0f}ms | "
            f"Tokens={tokens:>3} | "
            f"Question='{question[:30]}'"
        )

        # ── The last bot to finish prints the summary table ──────────────────
        with results_lock:
            if len(results) >= NUM_CLIENTS:
                log.info("\n" + "🔥" * 20 + " ALL BOTS COMPLETE " + "🔥" * 20)
                if len(results) == NUM_CLIENTS:
                    print_summary_table()

        raise StopUser()