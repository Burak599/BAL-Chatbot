import { checkHealth, loadAuthStatus } from './api.js';
import { clearChat, initializeChatApp, sendMessage, sendSuggestion } from './chat.js';

const ENTRY_ACCEPTED_KEY = 'bal_asistan_entry_accepted_v1';
const gateTabs = document.querySelectorAll('.gate-tab');
const gateSections = document.querySelectorAll('.gate-section');
const gateAgree = document.getElementById('gate-agree');
const gateContinue = document.getElementById('gate-continue');
const gateStatus = document.getElementById('gate-status');
const inputEl = document.getElementById('user-input');

const visitedGateTabs = {
  terms: true,
  about: false,
};

function openGateTab(tabName) {
  visitedGateTabs[tabName] = true;

  gateTabs.forEach((tab) => {
    const isActive = tab.dataset.tab === tabName;
    tab.classList.toggle('active', isActive);
    tab.classList.toggle('visited', visitedGateTabs[tab.dataset.tab]);
    tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });

  gateSections.forEach((section) => {
    section.classList.toggle('active', section.id === `panel-${tabName}`);
  });

  updateGateContinueState();
}

function updateGateContinueState() {
  if (!gateAgree || !gateContinue || !gateStatus) return;

  const canContinue = gateAgree.checked;
  gateContinue.disabled = !canContinue;
  gateStatus.textContent = canContinue
    ? 'Hazır. Devam ederek BAL Asistanı açabilirsin.'
    : 'Devam etmek için onay kutusunu işaretle.';
}

function enterChat() {
  document.body.classList.remove('gate-active');
  inputEl?.focus();
}

function initEntryGate() {
  localStorage.removeItem(ENTRY_ACCEPTED_KEY);

  gateTabs.forEach((tab) => {
    tab.addEventListener('click', () => openGateTab(tab.dataset.tab));
  });

  gateAgree?.addEventListener('change', updateGateContinueState);
  gateContinue?.addEventListener('click', () => {
    if (!gateContinue.disabled) enterChat();
  });

  updateGateContinueState();
}

function bootstrap() {
  initEntryGate();
  initializeChatApp();

  window.sendSuggestion = sendSuggestion;
  window.sendMessage = sendMessage;
  window.clearChat = clearChat;

  checkHealth();
  loadAuthStatus();
  setInterval(() => {
    checkHealth().catch(() => {});
  }, 30000);
}

bootstrap();
