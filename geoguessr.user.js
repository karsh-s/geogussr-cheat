// ==UserScript==
// @name         GeoCLIP Assistant
// @namespace    https://github.com/karsh-s/geogussr-cheat
// @version      1.0
// @description  Country prediction overlay driven by your local GeoCLIP model
// @match        https://www.geoguessr.com/*
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// @connect      localhost
// @run-at       document-start
// ==/UserScript==

/*
 * Pairs with geoguessr_server.py (run it first: python geoguessr_server.py).
 *
 * How it works: Street View tile requests all carry the panorama id in
 * their URL. A PerformanceObserver watches resource loads, pulls the id
 * out, and sends just that id to the local server — which fetches the
 * clean panorama itself, splits it into four compass headings, and runs
 * the model. Nothing is screenshotted, so the game UI never pollutes the
 * model input.
 *
 * Evidence accumulates across every panorama you visit in a round; press
 * "New round" to clear it.
 */

(function () {
  'use strict';

  const SERVER = 'http://127.0.0.1:8010';
  const seen = new Set();       // pano ids already sent
  let sums = new Map();         // country -> summed probability
  let panoCount = 0;
  let busy = false;
  const queue = [];

  // ── Overlay ──────────────────────────────────────────────
  let el = {};

  function buildUI() {
    const root = document.createElement('div');
    root.id = 'geoclip-panel';
    root.innerHTML = `
      <style>
        #geoclip-panel {
          position: fixed; top: 16px; left: 16px; z-index: 2147483647;
          width: 290px; background: rgba(20,20,24,.93); color: #f2f2f7;
          font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          border-radius: 12px; padding: 12px 14px; user-select: none;
          box-shadow: 0 8px 28px rgba(0,0,0,.45); backdrop-filter: blur(6px);
        }
        #geoclip-panel .gc-head { display:flex; justify-content:space-between;
          align-items:baseline; cursor:move; margin-bottom:2px; }
        #geoclip-panel .gc-country { font-size:20px; font-weight:650; }
        #geoclip-panel .gc-conf { font-size:12px; margin-bottom:8px; }
        #geoclip-panel .gc-row { display:flex; align-items:center; gap:6px;
          margin:3px 0; }
        #geoclip-panel .gc-name { width:104px; overflow:hidden;
          text-overflow:ellipsis; white-space:nowrap; font-size:12px; }
        #geoclip-panel .gc-bar { flex:1; height:9px; background:#3a3a3f;
          border-radius:5px; overflow:hidden; }
        #geoclip-panel .gc-fill { height:100%; background:#0a84ff; width:0%; }
        #geoclip-panel .gc-pct { width:42px; text-align:right; font-size:11px;
          color:#c7c7cc; }
        #geoclip-panel .gc-foot { display:flex; justify-content:space-between;
          align-items:center; margin-top:10px; font-size:11px; color:#8e8e93; }
        #geoclip-panel button { background:#2c2c2e; color:#f2f2f7; border:0;
          border-radius:7px; padding:4px 10px; font-size:11px; cursor:pointer; }
        #geoclip-panel button:hover { background:#3a3a3c; }
      </style>
      <div class="gc-head"><span class="gc-country">Waiting…</span></div>
      <div class="gc-conf"></div>
      <div class="gc-rows"></div>
      <div class="gc-foot">
        <span class="gc-status">connecting…</span>
        <button class="gc-reset">New round</button>
      </div>`;
    (document.body || document.documentElement).appendChild(root);

    el.root = root;
    el.country = root.querySelector('.gc-country');
    el.conf = root.querySelector('.gc-conf');
    el.rows = root.querySelector('.gc-rows');
    el.status = root.querySelector('.gc-status');
    root.querySelector('.gc-reset').onclick = reset;

    // draggable by the header
    const head = root.querySelector('.gc-head');
    let dx = 0, dy = 0, dragging = false;
    head.onmousedown = (e) => {
      dragging = true;
      dx = e.clientX - root.offsetLeft;
      dy = e.clientY - root.offsetTop;
      e.preventDefault();
    };
    document.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      root.style.left = (e.clientX - dx) + 'px';
      root.style.top = (e.clientY - dy) + 'px';
    });
    document.addEventListener('mouseup', () => { dragging = false; });
  }

  function reset() {
    sums = new Map();
    panoCount = 0;
    seen.clear();
    el.country.textContent = 'Waiting…';
    el.conf.textContent = '';
    el.rows.innerHTML = '';
    el.status.textContent = 'reset — move to load a panorama';
  }

  function render() {
    if (!panoCount) return;
    const ranked = [...sums.entries()]
      .map(([c, s]) => [c, s / panoCount])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const [best, p] = ranked[0];
    let colour = '#ff453a', label = 'low confidence';
    if (p >= 0.75) { colour = '#30d158'; label = 'high confidence'; }
    else if (p >= 0.45) { colour = '#ffd60a'; label = 'medium confidence'; }

    el.country.textContent = best;
    el.conf.textContent = `${(p * 100).toFixed(1)}% — ${label}`;
    el.conf.style.color = colour;
    el.rows.innerHTML = ranked.map(([c, v]) => `
      <div class="gc-row">
        <span class="gc-name">${c}</span>
        <span class="gc-bar"><span class="gc-fill" style="width:${(v * 100).toFixed(1)}%;
          background:${c === best ? colour : '#0a84ff'}"></span></span>
        <span class="gc-pct">${(v * 100).toFixed(1)}%</span>
      </div>`).join('');
    el.status.textContent = `${panoCount} panorama${panoCount > 1 ? 's' : ''}`;
  }

  // ── Server calls ─────────────────────────────────────────
  function post(path, body, cb, onErr) {
    GM_xmlhttpRequest({
      method: 'POST', url: SERVER + path,
      headers: { 'Content-Type': 'application/json' },
      data: JSON.stringify(body), timeout: 30000,
      onload: (r) => {
        try { cb(JSON.parse(r.responseText)); }
        catch (e) { onErr && onErr(e); }
      },
      onerror: onErr, ontimeout: onErr,
    });
  }

  function checkHealth() {
    GM_xmlhttpRequest({
      method: 'GET', url: SERVER + '/health', timeout: 5000,
      onload: (r) => {
        try {
          const j = JSON.parse(r.responseText);
          el.status.textContent = `${j.model} · ${j.countries} countries · ${j.device}`;
        } catch { el.status.textContent = 'server replied oddly'; }
      },
      onerror: () => { el.status.textContent = 'server offline — run geoguessr_server.py'; },
      ontimeout: () => { el.status.textContent = 'server not responding'; },
    });
  }

  function pump() {
    if (busy || !queue.length) return;
    busy = true;
    const pid = queue.shift();
    el.status.textContent = 'analysing…';
    post('/predict', { pano_id: pid }, (res) => {
      busy = false;
      if (res && res.top) {
        panoCount++;
        for (const [c, p] of res.top) sums.set(c, (sums.get(c) || 0) + p);
        render();
      } else {
        el.status.textContent = (res && res.error) || 'no prediction';
      }
      pump();
    }, () => {
      busy = false;
      el.status.textContent = 'server offline — run geoguessr_server.py';
      pump();
    });
  }

  function onPano(pid) {
    if (seen.has(pid)) return;
    seen.add(pid);
    if (queue.length < 3) queue.push(pid);   // don't backlog while panning
    pump();
  }

  // ── Panorama id sniffing ─────────────────────────────────
  const PANO_RE = /[?&]panoid=([A-Za-z0-9_\-]+)/;

  function watch() {
    try {
      const obs = new PerformanceObserver((list) => {
        for (const e of list.getEntries()) {
          const m = PANO_RE.exec(e.name);
          if (m) onPano(m[1]);
        }
      });
      obs.observe({ entryTypes: ['resource'] });
    } catch (e) {
      console.warn('[GeoCLIP] PerformanceObserver unavailable', e);
    }

    // Fallback: some tiles load via fetch rather than <img>
    const origFetch = window.fetch;
    if (origFetch) {
      window.fetch = function (...args) {
        try {
          const url = typeof args[0] === 'string' ? args[0] : args[0]?.url;
          const m = url && PANO_RE.exec(url);
          if (m) onPano(m[1]);
        } catch { /* ignore */ }
        return origFetch.apply(this, args);
      };
    }
  }

  // ── Boot ─────────────────────────────────────────────────
  watch();   // start sniffing immediately, before the UI exists
  const boot = () => { buildUI(); checkHealth(); };
  if (document.body) boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
