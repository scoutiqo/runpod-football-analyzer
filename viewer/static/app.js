window.SC_DATA_URL="/static/data.json";
window.SC_EVENTS_URL="/static/events.json";
window.SC_VIDEO_URL="/media/source.mp4";
const _fetch=window.fetch.bind(window);
window.fetch=(i,o)=>{
  if(typeof i==="string"){
    if(i.endsWith("data.json"))   i=window.SC_DATA_URL;
    if(i.endsWith("events.json")) i=window.SC_EVENTS_URL;
  }
  return _fetch(i,o);
};
// === SCOUTIQO OVERLAY VIEWER — STABLE v2 ===
(async () => {
	  // ------------------------------------------------------------------
  // SEND PASS LABEL TO SERVER
  // ------------------------------------------------------------------
  function sendPassLabel(frameIndex){
    fetch('/label_pass', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        clip: 'test_short',
        t: frameIndex,
        event: 'pass'
      })
    }).then(res => console.log("Label sent:", frameIndex))
      .catch(err => console.error("Label error:", err));
  }

  // ------------------------------------------------------------------
  // KEYBOARD SHORTCUT: press "P" to label a pass at the current frame
  // ------------------------------------------------------------------
  document.addEventListener('keydown', function (e) {
    if (e.key === 'p' || e.key === 'P') {
      const frame = window.currentFrame ?? 0;
      sendPassLabel(frame);
      alert("PASS labeled at frame: " + frame);
    }
  });

  console.log("🟢 Booting overlay viewer...");
    let currentFrame = 0;
  window.currentFrame = 0; // expose for debugging / keyboard 

  // Config
  const qs  = new URLSearchParams(location.search);
  const FPS = Number(qs.get("fps")) || 25; // override with ?fps=30 etc.
  const DATA_URL = (window && window.SC_DATA_URL) || "/static/data.json";

  // Load data
  let data = [];
  try {
    const res = await fetch(DATA_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (e) {
    console.error("❌ Failed to load data.json:", e);
    return;
  }
  console.log("✅ Loaded", data.length, "records");

  // Group by frame index "t"
  const byT = new Map();
  for (const r of data) {
    const t = (r.t | 0);
    if (!byT.has(t)) byT.set(t, []);
    byT.get(t).push(r);
  }

  // Wait for <video>
  function waitForVideo(sel="video", tries=200, ms=200){
    return new Promise((resolve,reject)=>{
      let n=0;
      const tick=()=>{
        const v=document.querySelector(sel);
        if (v) return resolve(v);
        if (++n>=tries) return reject(new Error("No <video> found after waiting."));
        setTimeout(tick, ms);
      };
      tick();
    });
  }

  let v;
  try { v = await waitForVideo(); }
  catch(e){ console.error("❌", e.message); return; }

  console.log("🎥 Video element found, setting up overlay...");

  // Canvas overlay
  const wrap = v.parentElement || document.body;
  if (getComputedStyle(wrap).position === "static") wrap.style.position = "relative";

  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, {
    position: "absolute",
    left: 0,
    top: 0,
    pointerEvents: "none",
  });
  wrap.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  function syncCanvasSize(){
    const vw = v.videoWidth  || v.clientWidth  || 1280;
    const vh = v.videoHeight || v.clientHeight || 720;
    canvas.width  = vw;
    canvas.height = vh;
    canvas.style.width  = (v.clientWidth  ? `${v.clientWidth}px`  : `${vw}px`);
    canvas.style.height = (v.clientHeight ? `${v.clientHeight}px` : `${vh}px`);
  }
  v.addEventListener("loadedmetadata", syncCanvasSize);
  window.addEventListener("resize", syncCanvasSize);
  syncCanvasSize();

  // Helpers
  const TEAM_COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444"];

  function isBall(r){
    if (r.type === "ball") return true;
    if (r.label === "ball") return true;
    if (r.cls === 1) return true; // common schema: 0 person, 1 ball
    return false;
  }

  function drawFrame(t){
    const dets = byT.get(t) || [];
    ctx.clearRect(0,0,canvas.width,canvas.height);

    for (const r of dets){
      const x = Number(r.x) || 0;
      const y = Number(r.y) || 0;

      if (isBall(r)){
        const R=6;
        ctx.beginPath();
        ctx.arc(x,y,R,0,Math.PI*2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#000000";
        ctx.stroke();
        continue;
      }

      const w = Number(r.w) || 20;
      const h = Number(r.h) || 40;
      const teamField = (r.team ?? r.tid ?? r.team_id ?? 0);
      const col = TEAM_COLORS[Math.abs(Number(teamField)) % TEAM_COLORS.length];

      ctx.lineWidth = 2;
      ctx.strokeStyle = col;
      ctx.strokeRect(x - w/2, y - h/2, w, h);

      const label = `${r.jersey ?? r.id ?? r.track_id ?? ""}`;
      if (label && label !== "undefined"){
        ctx.font = "14px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif";
        ctx.fillStyle = col;
        ctx.fillText(label, x + 8, y - 8);
      }
    }
  }

  function loop(){
    const frame = Math.round(v.currentTime * FPS);
    currentFrame = frame;
    window.currentFrame = frame;      // expose to keyboard handler
    drawFrame(frame);
    requestAnimationFrame(loop);}
})();
