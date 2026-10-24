const DEFAULTS = { server: 'http://127.0.0.1:8010', hotkey: 'g', autoShow: false };
const $ = (id) => document.getElementById(id);

function setStatus(text, cls) {
  const el = $('status');
  el.textContent = text;
  el.className = cls || '';
}

function ping() {
  setStatus('checking…');
  chrome.runtime.sendMessage({ type: 'health' }, (r) => {
    if (r && r.ok) {
      const d = r.data;
      const cue = d.cue_mode === 'scored' ? 'evidence scored per image'
        : d.cue_mode === 'typical' ? 'typical clues only' : 'no clues';
      setStatus(`${d.model} · ${d.mode} · ${d.countries} countries · ${cue}`, 'ok');
    } else {
      setStatus('server offline — start geoguessr_server.py', 'bad');
    }
  });
}

chrome.storage.sync.get(DEFAULTS, (s) => {
  $('server').value = s.server;
  $('hotkey').value = s.hotkey;
  $('autoShow').checked = !!s.autoShow;
  ping();
});

$('save').onclick = () => {
  chrome.storage.sync.set({
    server: ($('server').value || DEFAULTS.server).trim(),
    hotkey: ($('hotkey').value || 'g').trim().toLowerCase(),
    autoShow: $('autoShow').checked,
  }, () => {
    setStatus('saved — reload the game tab', 'ok');
    setTimeout(ping, 900);
  });
};
