"""
=============================================================
 BAL Chatbot — 20 Concurrent Bot Load Test
 Usage: python scripts/test_concurrent.py
=============================================================
This script:
  1. Spawns 20 bot threads simultaneously
  2. Each bot sends a random question to /api/chat
  3. Measures TTFT and total response time
  4. Prints a summary table when all bots finish
=============================================================
"""

import json
import random
import time
import uuid
import threading
import sys
import os
from datetime import datetime

# ZORLA buffer'sız output — her şey anında görünsün
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

import requests

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL = "https://brk9999-bal-chatbot.hf.space"
NUM_BOTS = 20
REQUEST_TIMEOUT = 180

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

results = []
results_lock = threading.Lock()
start_barrier = threading.Barrier(NUM_BOTS)  # Tüm botlar aynı anda başlasın


def bot_worker(bot_id: int):
    question = random.choice(QUESTIONS)
    fingerprint = f"test_bot_{bot_id}_{uuid.uuid4().hex[:8]}"
    session_id = f"session_bot_{bot_id}_{uuid.uuid4().hex[:8]}"

    headers = {
        "X-Client-Fingerprint": fingerprint,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    payload = {
        "message": question,
        "session_id": session_id,
    }

    # Tüm botlar burada bekler, son bot gelince hepsi aynı anda fırlar
    start_barrier.wait()

    start_time = time.time()
    first_token_at = None
    end_time = None
    tokens = 0
    error = None

    try:
        resp = requests.post(
            f"{BASE_URL}/api/chat",
            json=payload,
            headers=headers,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code != 200:
            error = f"HTTP {resp.status_code}"
            try:
                error += f" - {resp.text[:200]}"
            except:
                pass
        else:
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
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

    except requests.exceptions.Timeout:
        error = "TIMEOUT"
        end_time = time.time()
    except requests.exceptions.ConnectionError as e:
        error = f"CONNECTION_ERROR"
        end_time = time.time()
    except Exception as e:
        error = str(e)[:80]
        end_time = time.time()

    if end_time is None:
        end_time = time.time()

    duration = end_time - start_time
    ttft = first_token_at - start_time if first_token_at else None

    with results_lock:
        results.append({
            "bot_id": bot_id,
            "ttft": ttft,
            "duration": duration,
            "tokens": tokens,
            "question": question,
            "error": error,
        })

    status_icon = "OK" if not error else "ERR"
    ttft_str = f"{ttft*1000:.0f}ms" if ttft is not None else "  N/A  "
    line = f"{status_icon} Bot#{bot_id:>2} | TTFT={ttft_str:>7} | Total={duration*1000:>6.0f}ms | Tokens={tokens:>3} | {question[:30]}"
    print(line)
    sys.stdout.flush()


def main():
    # Başlık
    print("=" * 60)
    print("BAL Chatbot - 20 Concurrent Bot Load Test")
    print(f"  Server: {BASE_URL}")
    print(f"  Bots  : {NUM_BOTS}")
    print(f"  Time  : {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    print("  Testing connection...")
    sys.stdout.flush()

    # Health check
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=15)
        print(f"  Health: HTTP {r.status_code}")
        sys.stdout.flush()
    except Exception as e:
        print(f"  WARNING: Health check failed: {e}")
        print(f"  Trying anyway...")
        sys.stdout.flush()

    print("")
    print(f"  Spawning {NUM_BOTS} bots...")
    sys.stdout.flush()

    # Thread'leri başlat
    threads = []
    for i in range(1, NUM_BOTS + 1):
        t = threading.Thread(target=bot_worker, args=(i,))
        threads.append(t)
        t.start()

    print("  All bots started, waiting for results...")
    print("")
    sys.stdout.flush()

    # Bekle
    for t in threads:
        t.join()

    # Özet tablo
    print("")
    print("=" * 100)
    print("RESULTS TABLE")
    print("=" * 100)
    print(f"{'Bot#':<5} {'St':<4} {'TTFT(ms)':<12} {'Total(ms)':<14} {'Tok':<5} {'Question'}")
    print("-" * 100)
    sys.stdout.flush()

    total_ttft = []
    total_duration = []
    total_tokens = []

    for r in sorted(results, key=lambda x: x["bot_id"]):
        status = "OK" if not r["error"] else "ERR"
        ttft_ms = f"{r['ttft']*1000:.0f}" if r['ttft'] is not None else "N/A"
        dur_ms = f"{r['duration']*1000:.0f}"
        token_str = str(r["tokens"])

        if r["ttft"] is not None:
            total_ttft.append(r["ttft"])
        total_duration.append(r["duration"])
        total_tokens.append(r["tokens"])

        q_short = r["question"][:40]
        print(f"  {r['bot_id']:<3} {status:<4} {ttft_ms:<12} {dur_ms:<14} {token_str:<5} {q_short}")
        sys.stdout.flush()

    print("-" * 100)
    sys.stdout.flush()

    if total_ttft:
        avg_ttft = sum(total_ttft) / len(total_ttft)
        min_ttft = min(total_ttft)
        max_ttft = max(total_ttft)
    else:
        avg_ttft = min_ttft = max_ttft = 0.0

    avg_dur = sum(total_duration) / len(total_duration)
    min_dur = min(total_duration)
    max_dur = max(total_duration)
    avg_tok = sum(total_tokens) / len(total_tokens) if total_tokens else 0
    success = sum(1 for r in results if not r["error"])
    failed = sum(1 for r in results if r["error"])

    print("SUMMARY:")
    print(f"  Server      : {BASE_URL}")
    print(f"  OK/Total    : {success}/{len(results)}")
    print(f"  Failed      : {failed}")
    print(f"")
    print(f"  TTFT:")
    print(f"    Avg       : {avg_ttft*1000:.0f} ms")
    print(f"    Min       : {min_ttft*1000:.0f} ms")
    print(f"    Max       : {max_ttft*1000:.0f} ms")
    print(f"")
    print(f"  Total Time:")
    print(f"    Avg       : {avg_dur*1000:.0f} ms")
    print(f"    Min       : {min_dur*1000:.0f} ms")
    print(f"    Max       : {max_dur*1000:.0f} ms")
    print(f"")
    print(f"  Avg Tokens  : {avg_tok:.1f}")
    if avg_dur > 0:
        print(f"  Avg Tok/s   : {avg_tok/avg_dur:.1f}")
    print("=" * 100)
    sys.stdout.flush()


if __name__ == "__main__":
    main()