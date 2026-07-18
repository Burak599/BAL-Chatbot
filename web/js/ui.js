import { formatText } from './utils.js';

const messagesEl = document.getElementById('messages');
const noticeLayer = document.getElementById('notice-layer');

export function scrollToBottom() {
  if (messagesEl) {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
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

export function appendTypingIndicator() {
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

export function showLimitNotice(remaining = null, stateRef) {
  if (!noticeLayer || stateRef.limitNoticeVisible) return;

  stateRef.limitNoticeVisible = true;
  noticeLayer.innerHTML = '';

  const pill = document.createElement('div');
  pill.className = 'notice-pill';
  const countText = Number.isFinite(remaining) ? ` ${remaining} günlük hakkın kaldı.` : '';
  pill.innerHTML = `
    <span>Hakkın az kaldı.${countText}</span>
    <button class="notice-close" type="button" aria-label="Kapat">×</button>
  `;

  pill.querySelector('.notice-close')?.addEventListener('click', () => {
    stateRef.limitNoticeVisible = false;
    pill.remove();
  });

  noticeLayer.appendChild(pill);
}

export function showCongestionNotice(stateRef) {
  if (!noticeLayer || stateRef.congestionNoticeVisible) return;

  stateRef.congestionNoticeVisible = true;
  const pill = document.createElement('div');
  pill.className = 'notice-pill';
  pill.innerHTML = `
    <span>⚠️ Şu anda yoğunluk var, yanıtlar normalden geç gelebilir.</span>
    <button class="notice-close" type="button" aria-label="Kapat">×</button>
  `;

  pill.querySelector('.notice-close')?.addEventListener('click', () => {
    stateRef.congestionNoticeVisible = false;
    pill.remove();
  });

  noticeLayer.appendChild(pill);
}

export function hideCongestionNotice(stateRef) {
  if (!noticeLayer || !stateRef.congestionNoticeVisible) return;

  stateRef.congestionNoticeVisible = false;
  const pills = noticeLayer.querySelectorAll('.notice-pill');
  pills.forEach((pill) => {
    if (pill.textContent.includes('yoğunluk')) pill.remove();
  });
}

export function showModelFallbackNotice(notice, stateRef) {
  if (!noticeLayer) return;

  const targetModel = notice?.to_model || 'unknown';
  if (stateRef.notifiedFallbackModels.has(targetModel)) return;
  stateRef.notifiedFallbackModels.add(targetModel);

  const pill = document.createElement('div');
  pill.className = 'notice-pill';
  pill.innerHTML = `
    <span>Yoğunluk nedeniyle farklı bir model kullanılıyor.</span>
    <button class="notice-close" type="button" aria-label="Kapat">×</button>
  `;

  noticeLayer.appendChild(pill);
}
