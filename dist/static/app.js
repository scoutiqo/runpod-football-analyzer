(() => {
  const st = {
    data:null, i:0, playing:false, last:0,
    selectedId:null,
    perId: new Map(), // id -> {lastX_m,lastY_m,lastI, dist_m}
    videoReady:false
  };

  const cv = document.querySelector("canvas");
  const cx = cv.getContext("2d");
  const v  = document.getElementById("srcVideo");

  const btnPlay = document.getElementById("play");
  const btnPrev = document.getElementById("prev");
  const btnNext = document.getElementById("next");
  const jump    = document.getElementById("jump");
  const showPlayers = document.getElementById("showPlayers");
  const showBall    = document.getElementById("showBall");
  const showEvents  = document.getElementById("showEvents");
  const tLbl   = document.getElementById("time");

  function waitForDataThenBoot(){
    if (window.__DATA__) return boot(window.__DATA__);
    setTimeout(waitForDataThenBoot, 50);
  }

  function pxToMeters(xpx, ypx){
    const m = st.data.meta, fm = m.field_m;
    const nx = xpx / m.width;
    const ny = ypx / m.height;
    const xm = fm.xmin + nx * (fm.xmax - fm.xmin);
    const ym = fm.ymin + ny * (fm.ymax - fm.ymin);
    return [xm, ym];
  }

  function drawCircle(x,y,r){ cx.beginPath(); cx.arc(x,y,r,0,Math.PI*2); cx.fill(); }
  function drawTag(x,y, lines, emph=false){
    cx.save();
    cx.font = emph ? "bold 14px system-ui" : "12px system-ui";
    const pad=6, lh=14;
    const w = Math.max(...lines.map(t => cx.measureText(t).width)) + pad*2;
    const h = lh*lines.length + pad*2;
    cx.fillStyle = "rgba(0,0,0,0.55)";
    cx.fillRect(x, y-h-10, w, h);
    cx.fillStyle = "#fff";
    lines.forEach((t, i)=> cx.fillText(t, x+pad, y-h-10+pad+lh*(i+0.8)));
    cx.restore();
  }

  // Click to select nearest player
  cv.addEventListener("click", (e) => {
    if(!st.data) return;
    const r = cv.getBoundingClientRect();
    const x = (e.clientX - r.left) * (cv.width / r.width);
    const y = (e.clientY - r.top)  * (cv.height / r.height);
    const f = st.data.frames[st.i]; if(!f) return;
    let best = {id:null, d:1e9};
    for(const p of f.players){
      const d=Math.hypot(p.x-x, p.y-y);
      if(d<best.d){ best={id:p.id, d}; }
    }
    if(best.id!==null && best.d <= 30) st.selectedId = best.id;
  });

  function boot(d){
    st.data = d;
    cv.width  = d.meta.width;
    cv.height = d.meta.height;
    if (jump) jump.max = (d.frames.length-1).toString();

    if (v){
      v.pause(); v.muted = true; v.preload = "auto";
      v.addEventListener("loadeddata", ()=>{ st.videoReady=true; drawFrame(); }, {once:true});
      // kick decode on some browsers
      try { v.currentTime = 0; } catch {}
    }
    drawFrame();
  }

  // accumulate distance & compute smoothed speed
  function updateKinematics(fIdx){
    const f = st.data.frames[fIdx];
    const fps = st.data.meta.fps;
    const lag = Math.max(1, Math.floor(fps/5)); // ~5 Hz
    const prevIdx = Math.max(0, fIdx - lag);
    const fp = st.data.frames[prevIdx];

    const prevMap = new Map();
    for (const p of fp.players){
      const [xpm, ypm] = pxToMeters(p.x, p.y);
      prevMap.set(p.id, {xpm, ypm});
    }

    for (const p of f.players){
      const [xm, ym] = pxToMeters(p.x, p.y);

      let rec = st.perId.get(p.id);
      if(!rec){ rec = {lastX_m:xm, lastY_m:ym, lastI:fIdx, dist_m:0}; st.perId.set(p.id, rec); }
      const dt = (fIdx - rec.lastI) / fps;
      if (dt > 0){
        const dm = Math.hypot(xm - rec.lastX_m, ym - rec.lastY_m);
        if (dm < 20) rec.dist_m += dm; // guard teleport
        rec.lastX_m = xm; rec.lastY_m = ym; rec.lastI = fIdx;
      }

      const prev = prevMap.get(p.id);
      let v_ms = 0;
      if(prev){
        const dm = Math.hypot(xm - prev.xpm, ym - prev.ypm);
        const dtw = (fIdx - prevIdx) / fps;
        if (dtw > 0) v_ms = dm / dtw;
      }
      p._speed_kmh = v_ms * 3.6;
      p._dist_m    = rec.dist_m;
    }
  }

  function drawFrame(){
    if(!st.data) return;
    const f = st.data.frames[st.i]; if(!f) return;

    // 1) sync & draw video
    if (v && st.videoReady && !Number.isNaN(v.duration)){
      const tSec = st.i / st.data.meta.fps;
      if (Math.abs(v.currentTime - tSec) > (1 / st.data.meta.fps)){
        try { v.currentTime = tSec; } catch {}
      }
      try { cx.drawImage(v, 0, 0, cv.width, cv.height); }
      catch { cx.clearRect(0,0,cv.width,cv.height); }
    } else {
      cx.clearRect(0,0,cv.width,cv.height);
    }

    // 2) kinematics
    updateKinematics(st.i);

    // 3) players with broadcast-like labels
    for (const p of f.players){
      const selected = (p.id === st.selectedId);
      cx.fillStyle = selected
        ? "#ef4444"
        : (p.team===1 ? "#22c55e" : (p.team===0 ? "#3b82f6" : "#e5e7eb"));
      drawCircle(p.x, p.y, selected ? 9 : 7);

      const speed = Math.round(p._speed_kmh);
      const dist  = Math.round(p._dist_m);
      const lines = [`#${p.id}`, `${speed} km/h`, `${dist} m`];
      drawTag(p.x + 12, p.y - 12, lines, selected);
    }

    // 4) ball
    if (showBall?.checked && f.ball){
      cx.fillStyle = "#f59e0b";
      drawCircle(f.ball.x, f.ball.y, 6);
    }

    if (tLbl) tLbl.textContent = `t=${f.t} (${(st.i/st.data.meta.fps).toFixed(1)}s)`;
  }

  function step(n){
    if(!st.data) return;
    const N = st.data.frames.length;
    st.i = Math.max(0, Math.min(N-1, st.i + n));
    if (jump) jump.value = st.i;
    drawFrame();
  }
  function tick(ts){
    if(!st.playing){ st.last = ts; return; }
    const dt = ts - st.last;
    const ms = 1000 / st.data.meta.fps;
    if (dt >= ms) { step(1); st.last = ts; }
    requestAnimationFrame(tick);
  }

  btnPlay?.addEventListener("click", () => {
    st.playing = !st.playing;
    btnPlay.textContent = st.playing ? "Pause" : "Play";
    if (st.playing) requestAnimationFrame(tick);
  });
  btnPrev?.addEventListener("click", () => step(-1));
  btnNext?.addEventListener("click", () => step(+1));
  jump?.addEventListener("input", (e) => {
    st.i = Math.max(0, Math.min((st.data?.frames.length ?? 1)-1, parseInt(e.target.value || "0")));
    drawFrame();
  });
  [showPlayers, showBall, showEvents].forEach(el => el?.addEventListener("change", drawFrame));

  waitForDataThenBoot();
})();
