// ---- SCOUTIQO PATH PRELUDE (injected) ----
window.SC_DATA_URL   = "/static/data.json";
window.SC_EVENTS_URL = "/static/events.json";
window.SC_VIDEO_URL  = "/media/source.mp4";

// Force JSON fetches to our static files
const _fetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  try {
    if (typeof input === "string") {
      if (input.endsWith("data.json"))   input = window.SC_DATA_URL;
      if (input.endsWith("events.json")) input = window.SC_EVENTS_URL;
    }
  } catch(e){}
  return _fetch(input, init);
};

// Force the viewer's <video> element to use our media URL
(function ensureVideoSrc(){
  function setSrc(){
    const v = document.querySelector("video");
    if (v && v.src !== window.location.origin + window.SC_VIDEO_URL && v.getAttribute("src") !== window.SC_VIDEO_URL) {
      v.src = window.SC_VIDEO_URL;
      try { v.load(); } catch(e){}
    }
  }
  // Run now & on DOM changes
  setSrc();
  new MutationObserver(setSrc).observe(document.documentElement, { childList:true, subtree:true });
  window.addEventListener("load", setSrc);
})();
// ---- END PRELUDE ----
// ---- SCOUTIQO PATH PRELUDE (injected) ----
window.SC_DATA_URL   = "/static/data.json";
window.SC_EVENTS_URL = "/static/events.json";
window.SC_VIDEO_URL  = "/media/source.mp4";

// If app.js fetches "data.json"/"events.json" relatively, rewrite to absolute.
const _fetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  try {
    if (typeof input === "string") {
      if (input.endsWith("data.json"))   input = window.SC_DATA_URL;
      if (input.endsWith("events.json")) input = window.SC_EVENTS_URL;
    }
  } catch(e){}
  return _fetch(input, init);
};
// ---- END PRELUDE ----
const STATE = {
  meta: null,
  frames: [],
  idx: 0,
  playing: true,
  lastTs: 0,
  fps: 25,
};

const DATA_URLS = [
  'static/data_wrapped.json',
  'static/data.json'
];

function getCanvas() {
  // Try id first, then first canvas
  return document.getElementById('cv') || document.querySelector('canvas');
}

async function loadFirstAvailable(urls) {
  for (const u of urls) {
    try {
      const res = await fetch(u, { cache: 'no-store' });
      if (!res.ok) continue;
      const raw = await res.json();
      const isArray = Array.isArray(raw);
      const frames = isArray ? raw : (raw.frames || []);
      const meta = !isArray && raw.meta ? raw.meta : {
        fps: raw.fps || 25,
        width: raw.w || 1280,
        height: raw.h || 720,
        frames: frames.length
      };
      return { frames, meta, url: u };
    } catch (e) {
      // try next
    }
  }
  throw new Error('No data file found among: ' + urls.join(', '));
}

async function init() {
  const canvas = getCanvas();
  if (!canvas) {
    console.error('No <canvas> found (id="cv" or first <canvas>)');
    return;
  }
  const ctx = canvas.getContext('2d');

  try {
    const { frames, meta, url } = await loadFirstAvailable(DATA_URLS);
    STATE.frames = frames;
    STATE.meta = meta;
    STATE.fps = meta.fps || 25;

    // Resize canvas once we know meta
    canvas.width = meta.width || 1280;
    canvas.height = meta.height || 720;

    console.log('Loaded data from', url, 'frames=', frames.length, 'meta=', meta);
  } catch (err) {
    console.error('Failed to load data:', err);
  }

  requestAnimationFrame(tick);
}


function tick(ts) {
  const canvas = getCanvas();
  if (!canvas) return requestAnimationFrame(tick);
  const ctx = canvas.getContext('2d');

  // If not ready, keep waiting
  if (!STATE.meta || !STATE.frames || STATE.frames.length === 0) {
    ctx.fillStyle = '#0b1020';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#9ca3af';
    ctx.font = '16px system-ui, sans-serif';
    ctx.fillText('Loading data...', 24, 40);
    return requestAnimationFrame(tick);
  }

  const { meta, frames } = STATE;

  // Playback clock
  if (STATE.playing) {
    if (!STATE.lastTs) STATE.lastTs = ts;
    const dt = (ts - STATE.lastTs) / 1000.0;
    STATE.lastTs = ts;
    const step = Math.max(1, Math.floor(dt * (STATE.fps || 25)));
    STATE.idx = Math.min(STATE.idx + step, frames.length - 1);
  }

  // Resize canvas once in case meta changed
  if (canvas.width !== (meta.width || 1280) || canvas.height !== (meta.height || 720)) {
    canvas.width = meta.width || 1280;
    canvas.height = meta.height || 720;
  }

  // Clear
  ctx.fillStyle = '#0b1020';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw overlay header
  ctx.fillStyle = '#e2e8f0';
  ctx.font = '14px system-ui, sans-serif';
  ctx.fillText(`Frame ${STATE.idx+1}/${frames.length} @ ${meta.fps||25}fps`, 24, 40);

  // Entities
  const items = getFrameItems(frames[STATE.idx]);
  drawEntities(ctx, canvas, items);

  requestAnimationFrame(tick);
}
}

// Playback controls (wires up if the buttons exist)
function wireControls() {
  const playBtn = document.getElementById('playPause');
  const prevBtn = document.getElementById('prev');
  const nextBtn = document.getElementById('next');
  const jumpInput = document.getElementById('jump');

  if (playBtn) playBtn.onclick = () => { STATE.playing = !STATE.playing; playBtn.textContent = STATE.playing ? 'Pause' : 'Play'; };
  if (prevBtn) prevBtn.onclick = () => { STATE.idx = Math.max(0, STATE.idx - 1); };
  if (nextBtn) nextBtn.onclick = () => { STATE.idx = Math.min(STATE.frames.length - 1, STATE.idx + 1); };
  if (jumpInput) jumpInput.onchange = () => {
    const n = parseInt(jumpInput.value || '0', 10);
    if (!isNaN(n)) STATE.idx = Math.max(0, Math.min(STATE.frames.length - 1, n));
  };
}

window.addEventListener('DOMContentLoaded', () => {
  wireControls();
  init();
});
function getFrameItems(frame) {
  if (!frame) return [];
  if (Array.isArray(frame)) return frame;
  if (frame.items && Array.isArray(frame.items)) return frame.items;
  if (frame.objects && Array.isArray(frame.objects)) return frame.objects;
  return [];
}

function drawEntities(ctx, canvas, items) {
  // basic scale (if x,y are already in canvas space skip scaling)
  // We assume x,y are pixel coords within meta.width/height.
  // If your coords are normalized [0..1], uncomment the normalize lines.
  for (const it of items) {
    const cls = it.cls ?? it.class ?? 0;   // 0=player, 1=ball
    const x = it.x; // normalized? if so: const x = (it.x) * canvas.width;
    const y = it.y; // normalized? if so: const y = (it.y) * canvas.height;
    if (x == null || y == null) continue;

    if (cls === 1) {
      // ball
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI*2);
      ctx.fillStyle = '#ffd700';
      ctx.fill();
    } else {
      // player
      const team = it.team ?? -1;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI*2);
      ctx.fillStyle = (team === 0) ? '#60a5fa' : (team === 1 ? '#f87171' : '#a3a3a3');
      ctx.fill();

      // optional id label
      if (it.id != null) {
        ctx.fillStyle = '#e5e7eb';
        ctx.font = '10px system-ui, sans-serif';
        ctx.fillText(String(it.id), x+8, y-8);
      }
    }
  }
}

// ---- SCOUTIQO FORCED LOADER (appended) ----
;(function(){
  const video = document.querySelector('video');
  if (video && window.SC_VIDEO_URL) video.src = window.SC_VIDEO_URL;

  async function loadAll(){
    const [tracksRes, eventsRes] = await Promise.all([
      fetch(window.SC_DATA_URL),
      fetch(window.SC_EVENTS_URL).catch(()=>null)
    ]);
    const data = await tracksRes.json();
    const events = eventsRes && eventsRes.ok ? await eventsRes.json() : {events:[]};

    // Normalize into expected shape
    if (Array.isArray(data) || !data.meta) {
      console.error("Wrapped JSON missing or wrong shape.");
      return;
    }
    window.STATE = window.STATE || {};
    STATE.meta = data.meta;
    STATE.frames = data.frames || [];
    // kick any existing tick/init in the page
    if (typeof window.initViewer === "function") {
      window.initViewer(STATE, events);
    }
  }
  loadAll().catch(e=>console.error(e));
})();
 // ---- END FORCED LOADER ----
