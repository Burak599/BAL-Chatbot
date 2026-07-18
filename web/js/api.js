import { SESSION_ID, quotaInfo } from './state.js';

export { SESSION_ID };
export const API_BASE = window.location.origin + '/api';

const fingerprintPromise = import('/vendor/fingerprintjs/fp.esm.js').then((FingerprintJS) =>
  FingerprintJS.load({ monitoring: false })
);

let cachedFingerprint = '';

export function setQuotaFromStatus(data) {
  if (!data) return quotaInfo;

  quotaInfo.daily_used = Number.isFinite(data.daily_used) ? data.daily_used : 0;
  quotaInfo.minute_used = Number.isFinite(data.minute_used) ? data.minute_used : 0;
  quotaInfo.daily_limit = Number.isFinite(data.daily_limit) ? data.daily_limit : 40;
  quotaInfo.minute_limit = Number.isFinite(data.minute_limit) ? data.minute_limit : 5;

  return quotaInfo;
}

export async function getClientFingerprint() {
  if (cachedFingerprint) return cachedFingerprint;

  try {
    const fp = await fingerprintPromise;
    const result = await fp.get();
    cachedFingerprint = result.visitorId || '';
  } catch (err) {
    console.error('FingerprintJS yüklenemedi:', err);
  }

  return cachedFingerprint;
}

export async function buildApiHeaders(extraHeaders = {}) {
  const fingerprint = await getClientFingerprint();
  return {
    'Content-Type': 'application/json',
    ...(fingerprint ? { 'X-Client-Fingerprint': fingerprint } : {}),
    ...extraHeaders,
  };
}

export async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    credentials: 'same-origin',
    headers: await buildApiHeaders(options.headers || {}),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

export async function loadAuthStatus() {
  try {
    const data = await apiFetch('/auth/status', { method: 'GET', headers: {} });
    return setQuotaFromStatus(data);
  } catch {
    return quotaInfo;
  }
}
