"""
10 kullanıcı, aynı anda 1 soru sorar, cevap gelince durur.
Kullanım:
  locust -f locustfile.py --host https://chatbot-bal.onrender.com --users 10 --spawn-rate 10 --run-time 120s --headless
"""

import json
import random
import string
import time
import uuid
import logging
from locust import HttpUser, task, constant
from locust.exception import StopUser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bal_test")

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
]

class BALChatUser(HttpUser):
    wait_time = constant(0)  # bekleme yok

    def on_start(self):
        self.fingerprint = ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))
        self.client.headers.update({"X-Client-Fingerprint": self.fingerprint})
        self.done = False

    @task
    def send_one_question(self):
        if self.done:
            raise StopUser()

        self.done = True
        question = random.choice(QUESTIONS)
        
        # ⏱️ 1. AN: Sorunun gönderildiği tam an
        start_time = time.time()
        
        first_token_at = None
        end_time = None  # Cevabın tam bittiği anı tutacak
        tokens = 0
        error = None

        with self.client.post(
            "/api/chat",
            json={"message": question, "session_id": str(uuid.uuid4())},
            headers={"Accept": "text/event-stream"},
            stream=True,
            catch_response=True,
            timeout=120,
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
                        
                    # ⏱️ 2. AN: Sunucudan "bitti" işareti geldiği an
                    if data.get("done") or "__full_response__" in data:
                        end_type = "DONE_MARKER"
                        break
                
                # Eğer döngü bitti ama done yakalanamadıysa akışın kesildiği anı al
                if not end_time:
                    end_time = time.time()
                resp.success()

        # Eğer HTTP hatası vb. yüzünden with bloğundan erken çıkıldıysa end_time'ı garantiye al
        if end_time is None:
            end_time = time.time()

        # Net süre hesaplamaları
        total_duration = round(end_time - start_time, 2)
        ttft = round(first_token_at - start_time, 2) if first_token_at else None

        if error:
            log.info("❌ HATA     | süre=%ss | soru='%s' | hata=%s", total_duration, question[:40], error)
        else:
            log.info("✅ BAŞARILI | Soru Gönderim -> Cevap Bitiş Net Süre: %ss | TTFT=%ss | token=%d | soru='%s'",
                     total_duration, ttft, tokens, question[:40])

        raise StopUser()