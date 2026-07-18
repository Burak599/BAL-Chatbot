import { checkHealth, loadAuthStatus } from './api.js';
import { clearChat, initializeChatApp, sendMessage, sendSuggestion } from './chat.js';
import { initializeEntryGate } from './entry-gate.js';

function bootstrap() {
  initializeEntryGate();
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
