(async function(){
  const DATA_URL   = "/static/data.json";
  const EVENTS_URL = "/static/events.json";

  const cv         = document.getElementById('cv');
  const ctx        = cv.getContext('2d');
  const vid        = document.getElementById('vid');
  const playBtn    = document.getElementById('playBtn');
  const frameBox   = document.getElementById('frameBox');
  const chkPlayers = document.getElementById('chkPlayers');
  const chkBall    = document.getElementById('chkBall');
  const hud        = document.getElementById('hud');

  // Load artifacts
  const [tracksRes, eventsRes] = await Promise.all([
    fetch(DATA_URL), fetch(EVENTS_URL).catch(()=>null)
  ]);
  const data   = await tracksRes.json().catch(()=>null);
  const events = eventsRes && eventsRes.ok ? await eventsRes.json() : { events: [] };

  if (!data || !data.meta) {
    ctx.fillStyle="#e2e8f0"; ctx.fillText("Cannot load data.json", 40, 60);
    return;
  }

  // Your meta shows {"frames":203172,"fps":25,"schema":"frames_v1"}.
  // Accept both: (A) top-level frames array, or (B) nested under data.frames.
  const meta = data.meta || {};
  const fps  = Number(meta.fps || 25);
  let frames = Array.isArray(data.frames) ? data.frames : data.data || data.list || [];

  // Some exports put frame count in meta.frames (number) AND the array also named "frames".
  // If frames is not an array, try to detect an array-like key at top level.
  if (!Array.isArray(frames)) {
    for (const k of Object.keys(data)) {
      if (Array.isArray(data[k]) && data[k].length && data[k][0] && typeof data[k][0] === 'object' && ('t' in data[k][0])) {
        frames = data[k]; break;
      }
    }
  }

  if (!Array.isArray(frames) || frames.length === 0) {
    ctx.fillStyle="#e2e8f0";
    ctx.fillText("No per-frame data found in data.json", 40, 60);
    console.error("data.json keys:", Object.keys(data));
    return;
  }

  // Source dimensions; use your typical export if not known
  const srcW = Number(meta.w || 1280);
  const srcH = Number(meta.h || 720);

  function sizeToVideoOrMeta() {
    const vw = vid && vid.videoWidth  ? vid.videoWidth  : srcW;
    const vh = vid && vid.videoHeight ? vid.videoHeight : srcH;
    if (!vw || !vh) { cv.width = srcW; cv.height = srcH; return; }
    const W = Math.max(640, vw);
    const H = Math.round(W * vh / vw);
    cv.width  = W;
    cv.height = H;
  }
  sizeToVideoOrMeta();
  if (vid) vid.addEventListener('loadedmetadata', sizeToVideoOrMeta);

  // Controls
  if (playBtn) playBtn.addEventListener('click', () => {
    if (vid && vid.paused) { vid.play().catch(()=>{}); playBtn.textContent = "Pause"; }
    else if (vid) { vid.pause(); playBtn.textContent = "Play"; }
  });

  let lastFi = -1;
  function drawFrame(fi){
    const f = frames[fi] || {players:[], ball:null};

    // paint bg (video if available; else solid)
    ctx.fillStyle = "#0b1020";
    ctx.fillRect(0,0,cv.width,cv.height);
    try { if (vid && vid.readyState >= 2) ctx.drawImage(vid, 0, 0, cv.width, cv.height); } catch(e){}

    // scale factors
    const vw = (vid && vid.videoWidth)  ? vid.videoWidth  : srcW;
    const vh = (vid && vid.videoHeight) ? vid.videoHeight : srcH;
    const sx = cv.width / (vw||srcW);
    const sy = cv.height/ (vh||srcH);

    // players
    if (chkPlayers && chkPlayers.checked && Array.isArray(f.players)) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#7dd3fc";
      ctx.fillStyle   = "#7dd3fc";
      for (const p of f.players) {
        const x = (p.x||0) * sx, y = (p.y||0) * sy;
        ctx.strokeRect(x-6, y-6, 12, 12);
        ctx.font = "10px system-ui";
        ctx.fillText(String(p.id ?? ""), x+8, y-8);
      }
    }

    // ball
    if (chkBall && chkBall.checked && f.ball && Number.isFinite(f.ball.x) && Number.isFinite(f.ball.y)) {
      const bx = f.ball.x * sx, by = f.ball.y * sy;
      ctx.beginPath(); ctx.fillStyle = "#fbbf24";
      ctx.arc(bx, by, 4, 0, Math.PI*2); ctx.fill();
    }

    // HUD
    if (hud) hud.textContent = `frame:${fi+1}/${frames.length} | fps:${fps} | events:${events.events?.length||0}`;
  }

  function tick(){
    // if video exists, sync to time; else follow the input box
    let fi = 0;
    if (vid && fps) fi = Math.min(frames.length-1, Math.max(0, Math.floor((vid.currentTime||0)*fps)));
    const manual = parseInt(frameBox?.value||"1",10);
    if (!vid || vid.paused) fi = Math.min(frames.length-1, Math.max(0, (manual-1)));

    if (fi !== lastFi) {
      drawFrame(fi);
      if (frameBox) frameBox.value = String(fi+1);
      lastFi = fi;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  if (frameBox) frameBox.addEventListener('change', ()=>{
    const val = Math.max(1, Math.min(frames.length, parseInt(frameBox.value||"1",10)));
    frameBox.value = String(val);
    if (vid) vid.currentTime = (val-1)/fps;
    drawFrame(val-1);
  });

})();
