

# --- monitor2 overlay viewer (safe, no decorators) ---
def _monitor2_endpoint(job_id: str):
    import json
    from fastapi.responses import HTMLResponse
    job_js = json.dumps(job_id)
    html = r"""<!doctype html><meta charset="utf-8"/><title>Monitor</title>
<style>
  body{font-family:system-ui;margin:20px}
  .row{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
  video{width:640px;height:360px;background:#000;border-radius:8px}
  pre{max-height:520px;overflow:auto;background:#111;color:#eee;padding:12px;border-radius:8px;min-width:420px}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef;margin-left:8px}
  ul#arts{margin:8px 0 0 0;padding:0;list-style:none}
  ul#arts li{margin:4px 0}
  ul#arts a{text-decoration:none}
</style>
<h1>Job <code>__JOB_TEXT__</code> <span id="badge" class="pill">waiting…</span></h1>
<div class="row">
  <div>
    <video id="vid" controls muted playsinline></video>
    <div style="margin-top:8px">
      <button onclick="forcePoll()">Refresh now</button>
      <a id="dump" target="_blank">Open full event dump</a>
    </div>
    <div style="margin-top:12px">
      <h3 style="margin:0 0 6px">Artifacts</h3>
      <ul id="arts"></ul>
    </div>
  </div>
  <pre id="log"></pre>
</div>
<script>
const job   = __JOB_JSON__;
const base  = location.origin;
const logEl = document.getElementById('log');
const vid   = document.getElementById('vid');
const badge = document.getElementById('badge');
const dumpA = document.getElementById('dump');
const artsUl= document.getElementById('arts');
dumpA.href  = `${base}/progress/${job}/dump`;

let lastCount = 0;
let lastVideoSrc = "";

// prefer overlay videos if present
function pickBestVideoArtifact(arts) {
  const vids = arts.filter(a => /\.(mp4|webm)(\?|$)/i.test(a.url));
  if (vids.length === 0) return null;
  vids.sort((a,b) => (b.name && b.name.includes('overlay') ? 1 : 0) - (a.name && a.name.includes('overlay') ? 1 : 0));
  return vids.pop();
}

async function poll() {
  try {
    const st = await fetch(`${base}/status/${job}`).then(r => r.json());
    badge.textContent = st.last_type ? st.last_type : "waiting…";

    if ((st.events || 0) !== lastCount) {
      lastCount = st.events || 0;

      const dump = await fetch(`${base}/progress/${job}/dump`).then(r => r.json());
      logEl.textContent = JSON.stringify(dump, null, 2);
      const evts = (dump && dump.events) ? dump.events : [];

      const arts = [];
      for (const e of evts) {
        if (e.type === "artifact" && e.url) {
          arts.push({ name: e.name || "artifact", url: e.url, seg: e.seg });
        }
      }
      artsUl.innerHTML = "";
      for (const a of arts) {
        const li = document.createElement('li');
        const segText = (a.seg != null ? ('seg ' + a.seg + ': ') : '');
        li.innerHTML = `<code>${segText}</code><a href="${a.url}" target="_blank">${a.name || 'artifact'}</a>`;
        artsUl.appendChild(li);
      }

      const best = pickBestVideoArtifact(arts);
      if (best && best.url !== lastVideoSrc) {
        lastVideoSrc = best.url;
        vid.src = best.url;
        try { await vid.play(); } catch (e) {}
      } else {
        const latest = evts.length ? evts[evts.length - 1] : null;
        if (latest && latest.type === "segment_start" && latest.url && latest.url !== lastVideoSrc) {
          lastVideoSrc = latest.url;
          vid.src = latest.url;
          try { await vid.play(); } catch (e) {}
        }
      }
    }
  } catch (e) { badge.textContent = "error"; }
}
function forcePoll(){ lastCount = -1; poll(); }
setInterval(poll, 1000); poll();
</script>
"""
    html = html.replace("__JOB_JSON__", job_js).replace("__JOB_TEXT__", job_id)
    return HTMLResponse(html)

# Register route AFTER app exists
try:
    app.add_api_route("/monitor2/{job_id}", _monitor2_endpoint, methods=["GET"])
except Exception:
    # if app not yet defined at import time, a later import can re-run this or you can move these lines lower
    pass
# --- end monitor2 ---

