import { API_BASE } from './api.js';

export function sendFeedback(questionIndex, type) {
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

export function toggleFeedbackBox(questionIndex) {
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

export function submitFeedbackText(questionIndex) {
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

export function createFeedbackBar(questionIndex) {
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
