export const SESSION_ID = 'session_' + Math.random().toString(36).slice(2, 9);

export const quotaInfo = {
  daily_used: 0,
  minute_used: 0,
  daily_limit: 40,
  minute_limit: 5,
};

export const appState = {
  isStreaming: false,
  lastQuestionIndex: null,
  limitNoticeVisible: false,
  congestionNoticeVisible: false,
  notifiedFallbackModels: new Set(),
};
