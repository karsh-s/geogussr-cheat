// Service worker: the only part allowed to talk to the local server.
// Content scripts cannot make cross-origin requests to localhost, but a
// worker holding host_permissions can, so content.js messages us instead.

const DEFAULTS = { server: 'http://127.0.0.1:8010', hotkey: 'g', autoShow: false };

async function settings() {
  const s = await chrome.storage.sync.get(DEFAULTS);
  return { ...DEFAULTS, ...s };
}

async function call(path, body) {
  const { server } = await settings();
  const url = server.replace(/\/+$/, '') + path;
  const res = await fetch(url, body ? {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  } : {});
  if (!res.ok) throw new Error(`server returned ${res.status}`);
  return res.json();
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  const done = (ok, data) => reply(ok ? { ok: true, data } : { ok: false, error: String(data) });

  if (msg.type === 'health') {
    call('/health').then(d => done(true, d)).catch(e => done(false, e.message));
    return true;                      // keep the channel open for async reply
  }
  if (msg.type === 'predict') {
    call('/predict', { pano_id: msg.panoId })
      .then(d => done(true, d)).catch(e => done(false, e.message));
    return true;
  }
  if (msg.type === 'settings') {
    settings().then(s => done(true, s));
    return true;
  }
});
