import { API_BASE, buildApiHeaders, loadAuthStatus } from './api.js';
import { SESSION_ID, appState, quotaInfo } from './state.js';
import {
  appendMessage,
  appendTypingIndicator,
  scrollToBottom,
  showCongestionNotice,
  showLimitNotice,
  showModelFallbackNotice,
} from './ui.js';
import { createFeedbackBar } from './feedback.js';
import { formatText, stripReasoningText } from './utils.js';

const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const charCount = document.getElementById('char-count');

function updateQuotaDisplay() {
  const dailyRemaining = Math.max(quotaInfo.daily_limit - quotaInfo.daily_used, 0);
  const minuteRemaining = Math.max(quotaInfo.minute_limit - quotaInfo.minute_used, 0);

  if (dailyRemaining <= 10 || minuteRemaining <= 1) {
    showLimitNotice(dailyRemaining, appState);
  }
}

export async function sendMessage() {
  if (!inputEl || !sendBtn || !charCount) return;

  const message = inputEl.value.trim();
  if (!message || appState.isStreaming) return;

  appState.isStreaming = true;
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
      if (data.near_limit) showLimitNotice(null, appState);
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
    const messagesEl = document.getElementById('messages');
    messagesEl?.appendChild(wrap);

    const reader = response.body?.getReader();
    if (!reader) throw new Error('Akış yanıtı desteklenmiyor.');

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
          showCongestionNotice(appState);
        }

        if (data.model_fallback) {
          showModelFallbackNotice(data.model_fallback, appState);
        }

        if (data.done && data.question_index) {
          appState.lastQuestionIndex = data.question_index;
          wrap.appendChild(createFeedbackBar(appState.lastQuestionIndex));
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
      updateQuotaDisplay();
    }
  } catch (err) {
    typingWrap?.remove();
    const errorText = err.message || 'Teknik bir sorun oluştu. Lütfen daha sonra tekrar deneyin.';
    appendMessage('bot', errorText);
  }

  appState.isStreaming = false;
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

  const messagesEl = document.getElementById('messages');
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
