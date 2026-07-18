import { API_BASE, SESSION_ID, apiFetch, buildApiHeaders, loadAuthStatus, quotaInfo, setQuotaFromStatus } from './api.js';

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const charCount = document.getElementById('char-count');
const noticeLayer = document.getElementById('notice-layer');

let isStreaming = false;
let lastQuestionIndex = null;
let limitNoticeVisible = false;
let congestionNoticeVisible = false;
const notifiedFallbackModels = new Set();

function scrollToBottom() {
  if (messagesEl) {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

function updateQuotaDisplay() {
  const dailyRemaining = Math.max(quotaInfo.daily_limit - quotaInfo.daily_used, 0);
  const minuteRemaining = Math.max(quotaInfo.minute_limit - quotaInfo.minute_used, 0);

  if (dailyRemaining <= 10 || minuteRemaining <= 1) {
    showLimitNotice(dailyRemaining);
  }
}

function showLimitNotice(remaining = null) {
  if (!noticeLayer || limitNoticeVisible) return;

  limitNoticeVisible = true;
  noticeLayer.innerHTML = '';

  const pill = document.createElement('div');
  pill.className = 'notice-pill';
  const countText = Number.isFinite(remaining) ? ` ${remaining} günlük hakkın kaldı.` : '';
  pill.innerHTML = `
    <span>Hakkın az kaldı.${countText}</span>
    <button class="notice-close" type="button" aria-label="Kapat">×</button>
  `;

  pill.querySelector('.notice-close')?.addEventListener('click', () => {
    limitNoticeVisible = false;
    pill.remove();
  });

  noticeLayer.appendChild(pill);
}

function showCongestionNotice() {
  if (!noticeLayer || congestionNoticeVisible) return;

  congestionNoticeVisible = true;
  const pill = document.createElement('div');
  pill.className = 'notice-pill';
  pill.innerHTML = `
    <span>⚠️ Şu anda yoğunluk var, yanıtlar normalden geç gelebilir.</span>
    <button class="notice-close" type="button" aria-label="Kapat">×</button>
  `;

  pill.querySelector('.notice-close')?.addEventListener('click', () => {
    congestionNoticeVisible = false;
    pill.remove();
  });

  noticeLayer.appendChild(pill);
}

function hideCongestionNotice() {
  if (!noticeLayer || !congestionNoticeVisible) return;

  congestionNoticeVisible = false;
  const pills = noticeLayer.querySelectorAll('.notice-pill');
  pills.forEach((pill) => {
    if (pill.textContent.includes('yoğunluk')) pill.remove();
  });
}

function showModelFallbackNotice(notice) {
  if (!noticeLayer) return;

  const targetModel = notice?.to_model || 'unknown';
  if (notifiedFallbackModels.has(targetModel)) return;
  notifiedFallbackModels.add(targetModel);

  const pill = document.createElement('div');
  pill.className = 'notice-pill';
  pill.innerHTML = `
    <span>Yoğunluk nedeniyle farklı bir model kullanılıyor.</span>
    <button class="notice-close" type="button" aria-label="Kapat">×</button>
  `;

  noticeLayer.appendChild(pill);
}

export function appendMessage(role, text) {
  document.getElementById('welcome-msg')?.remove();

  const wrap = document.createElement('div');
  wrap.className = `msg-wrap ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = formatText(text);

  wrap.appendChild(bubble);
  messagesEl?.appendChild(wrap);
  scrollToBottom();
  return bubble;
}

function appendTypingIndicator() {
  document.getElementById('welcome-msg')?.remove();

  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap bot';
  wrap.id = 'typing-wrap';

  const indicator = document.createElement('div');
  indicator.className = 'typing-indicator';
  indicator.innerHTML = '<span></span><span></span><span></span>';

  wrap.appendChild(indicator);
  messagesEl?.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function formatText(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
    .replace(/^#{1,3}\s+(.+)$/gm, '<strong>$1</strong>')
    .replace(/^[-•]\s+(.+)$/gm, '• $1')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>')
    .replace(/^/, '<p>')
    .replace(/$/, '</p>');
}

function stripReasoningText(text) {
  return text
    .replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, '')
    .replace(/<thinking\b[^>]*>[\s\S]*?<\/thinking>/gi, '')
    .replace(/<think\b[^>]*>[\s\S]*$/gi, '')
    .replace(/<thinking\b[^>]*>[\s\S]*$/gi, '')
    .trim();
}

function sendFeedback(questionIndex, type) {
  const btns = document.querySelectorAll(`[data-qidx="${questionIndex}"] .fb-btn`);
  btns.forEach((button) => {
    const feedbackType = button.dataset.fbType;
    button.classList.toggle('fb-active', feedbackType === type);
  });

  fetch(`${API_BASE}/chat/feedback`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question_index: questionIndex, feedback: type }),
  }).catch(() => {});
}

function toggleFeedbackBox(questionIndex) {
  const box = document.querySelector(`[data-qidx="${questionIndex}"] .fb-text-box`);
  if (!box) return;

  const isVisible = box.style.display === 'flex';
  box.style.display = isVisible ? 'none' : 'flex';

  if (!isVisible) {
    const textarea = box.querySelector('textarea');
    if (textarea) {
      textarea.value = '';
      textarea.focus();
    }
  }
}

function submitFeedbackText(questionIndex) {
  const box = document.querySelector(`[data-qidx="${questionIndex}"] .fb-text-box`);
  if (!box) return;

  const textarea = box.querySelector('textarea');
  const text = textarea ? textarea.value.trim() : '';
  if (!text) {
    toggleFeedbackBox(questionIndex);
    return;
  }

  fetch(`${API_BASE}/chat/feedback`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question_index: questionIndex, feedback_text: text }),
  }).catch(() => {});

  box.style.display = 'none';

  const bar = box.closest('.fb-bar');
  if (bar) {
    const existing = bar.querySelector('.fb-thanks');
    if (existing) existing.remove();

    const thanks = document.createElement('div');
    thanks.className = 'fb-thanks';
    thanks.textContent = 'Geri bildiriminiz için teşekkür ederiz.';
    bar.appendChild(thanks);
    setTimeout(() => {
      thanks.remove();
    }, 2500);
  }
}

function createFeedbackBar(questionIndex) {
  const bar = document.createElement('div');
  bar.className = 'fb-bar';
  bar.dataset.qidx = questionIndex;

  const likeBtn = document.createElement('button');
  likeBtn.className = 'fb-btn';
  likeBtn.dataset.fbType = 'like';
  likeBtn.title = 'Yararlı';
  likeBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>`;
  likeBtn.addEventListener('click', () => sendFeedback(questionIndex, 'like'));

  const dislikeBtn = document.createElement('button');
  dislikeBtn.className = 'fb-btn';
  dislikeBtn.dataset.fbType = 'dislike';
  dislikeBtn.title = 'Yanlış veya yetersiz';
  dislikeBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V7H7.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10zM17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>`;
  dislikeBtn.addEventListener('click', () => sendFeedback(questionIndex, 'dislike'));

  const chatBtn = document.createElement('button');
  chatBtn.className = 'fb-btn';
  chatBtn.title = 'Geri bildirim yaz';
  chatBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;
  chatBtn.addEventListener('click', () => toggleFeedbackBox(questionIndex));

  bar.appendChild(likeBtn);
  bar.appendChild(dislikeBtn);
  bar.appendChild(chatBtn);

  const textBox = document.createElement('div');
  textBox.className = 'fb-text-box';
  textBox.style.display = 'none';
  textBox.innerHTML = `
    <textarea class="fb-textarea" placeholder="Geri bildiriminizi yazın..." rows="2" maxlength="500"></textarea>
    <div class="fb-text-actions">
      <button class="fb-text-cancel" type="button">İptal</button>
      <button class="fb-text-send" type="button">Gönder</button>
    </div>
  `;

  textBox.querySelector('.fb-text-send')?.addEventListener('click', () => submitFeedbackText(questionIndex));
  textBox.querySelector('.fb-text-cancel')?.addEventListener('click', () => {
    textBox.style.display = 'none';
  });
  textBox.querySelector('textarea')?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submitFeedbackText(questionIndex);
    }
  });

  bar.appendChild(textBox);
  return bar;
}

export async function sendMessage() {
  if (!inputEl || !sendBtn || !charCount) return;

  const message = inputEl.value.trim();
  if (!message || isStreaming) return;

  isStreaming = true;
  sendBtn.disabled = true;
  inputEl.value = '';
  inputEl.style.height = 'auto';
  charCount.textContent = '0 / 500';

  appendMessage('user', message);
  const typingWrap = appendTypingIndicator();

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: await buildApiHeaders(),
      body: JSON.stringify({ message, session_id: SESSION_ID }),
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      if (data.near_limit) showLimitNotice();
      const errorType = data.error_type || 'technical';
      const userMsg = errorType === 'quota' ? (data.error || 'Limit aşıldı.') : 'Teknik bir sorun oluştu. Lütfen daha sonra tekrar deneyin.';
      throw new Error(userMsg);
    }

    typingWrap.remove();

    const wrap = document.createElement('div');
    wrap.className = 'msg-wrap bot';
    wrap.style.animation = 'fadeUp .2s ease both';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = '<p></p>';

    wrap.appendChild(bubble);
    messagesEl?.appendChild(wrap);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let rawText = '';
    let sseBuffer = '';
    let sseHadError = false;

    const processSseLine = (line) => {
      if (!line.startsWith('data: ')) return;

      try {
        const data = JSON.parse(line.slice(6));

        if (data.token) {
          rawText += data.token;
          bubble.innerHTML = formatText(stripReasoningText(rawText));
          scrollToBottom();
        }

        if (data.error) {
          const errorType = data.error_type || 'technical';
          bubble.innerHTML = errorType === 'quota'
            ? `<p style="color:#d97706">${data.error}</p>`
            : '<p style="color:#667085">Teknik bir sorun oluştu. Lütfen daha sonra tekrar deneyin.</p>';
          sseHadError = true;
        }

        if (data.congestion) {
          showCongestionNotice();
        }

        if (data.model_fallback) {
          showModelFallbackNotice(data.model_fallback);
        }

        if (data.done && data.question_index) {
          lastQuestionIndex = data.question_index;
          const feedbackBar = createFeedbackBar(lastQuestionIndex);
          wrap.appendChild(feedbackBar);
        }
      } catch {
        // Ignore incomplete or malformed SSE fragments.
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        sseBuffer += decoder.decode();
        break;
      }

      sseBuffer += decoder.decode(value, { stream: true });
      const lines = sseBuffer.split('\n');
      sseBuffer = lines.pop() || '';

      lines.forEach((line) => processSseLine(line));
    }

    if (sseBuffer.trim()) {
      processSseLine(sseBuffer.trim());
    }

    if (!sseHadError) {
      await loadAuthStatus();
      setQuotaFromStatus(quotaInfo);
      updateQuotaDisplay();
    }
  } catch (err) {
    typingWrap?.remove();
    const errorText = err.message || 'Teknik bir sorun oluştu. Lütfen daha sonra tekrar deneyin.';
    appendMessage('bot', errorText);
  }

  isStreaming = false;
  sendBtn.disabled = false;
  inputEl?.focus();
  scrollToBottom();
}

export function sendSuggestion(element) {
  if (!inputEl) return;
  inputEl.value = element?.textContent?.trim() || '';
  sendMessage();
}

function welcomeMarkup(title, description, includeSuggestions = true) {
  const suggestions = includeSuggestions ? `
    <div class="suggestion-chips">
      <button class="chip" onclick="sendSuggestion(this)">LGS taban puanı nedir?</button>
      <button class="chip" onclick="sendSuggestion(this)">Ayran Günü nedir?</button>
      <button class="chip" onclick="sendSuggestion(this)">Okula nasıl kayıt yapılır?</button>
      <button class="chip" onclick="sendSuggestion(this)">BALEV bursu hakkında bilgi ver</button>
      <button class="chip" onclick="sendSuggestion(this)">Okula nasıl giderim?</button>
      <button class="chip" onclick="sendSuggestion(this)">YKS başarıları nasıl?</button>
    </div>` : '';

  return `
    <div class="welcome" id="welcome-msg" style="animation: fadeUp .3s ease both">
      <div class="icon">
        <img src="BAL_Logo.png" alt="BAL" />
      </div>
      <h2>${title}</h2>
      <p>${description}</p>
      ${suggestions}
    </div>`;
}

export async function clearChat() {
  try {
    await fetch(`${API_BASE}/clear`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: await buildApiHeaders(),
      body: JSON.stringify({ session_id: SESSION_ID }),
    });
  } catch {
    // Continue silently even if the reset request fails.
  }

  if (messagesEl) {
    messagesEl.innerHTML = welcomeMarkup(
      'Sohbet sıfırlandı',
      'Yeni bir BAL sorusuyla devam edebilirsin.'
    );
  }
}

export function initializeChatApp() {
  updateQuotaDisplay();

  if (!inputEl || !sendBtn || !charCount) return;

  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 140)}px`;
    charCount.textContent = `${inputEl.value.length} / 500`;
  });

  inputEl.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });

  loadAuthStatus().then(() => {
    updateQuotaDisplay();
  });
}
