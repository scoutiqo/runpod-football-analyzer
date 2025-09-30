from typing import Dict, Any, Optional
from fastapi import APIRouter, Body
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/passnet", tags=["passnet"])

# ---------------- Health ----------------
@router.get("/health")
def passnet_health():
    return {"ok": True, "repo": "runpod-football-analyzer"}

# -------------- In-memory state ----------
PASS_STATE: Dict[str, Dict[str, Any]] = {}  # {job_id: {"teams": {tid: {"nodes":{id:node}, "links":{(s,t):w}}}}}

def _ensure_job(job_id: str):
    if job_id not in PASS_STATE:
        PASS_STATE[job_id] = {"teams": {}}
    return PASS_STATE[job_id]

def _ensure_team(job: Dict[str, Any], team_id: int):
    teams = job["teams"]
    if team_id not in teams:
        teams[team_id] = {"team_id": team_id, "nodes": {}, "links": {}}
    return teams[team_id]

def _ensure_node(team: Dict[str, Any], tid: int, label: Optional[str]=None, x: float=0.0, y: float=0.0):
    nodes = team["nodes"]
    if tid not in nodes:
        nodes[tid] = {"id": tid, "label": str(label if label is not None else tid), "x": float(x), "y": float(y)}
    return nodes[tid]

# ---------------- Live API ----------------
@router.get("/live/{job_id}")
def passnet_live(job_id: str):
    job = _ensure_job(job_id)
    out = []
    for tid, team in job["teams"].items():
        nodes = list(team["nodes"].values())
        links = [{"source": s, "target": t, "weight": w} for (s,t), w in team["links"].items()]
        out.append({"team_id": tid, "nodes": nodes, "links": links})
    return {"job_id": job_id, "teams": out}

@router.post("/live/{job_id}/add")
def passnet_add(job_id: str,
                team_id: int = Body(...),
                source: int = Body(...),
                target: int = Body(...),
                sx: float = Body(0.0), sy: float = Body(0.0),
                tx: float = Body(0.0), ty: float = Body(0.0),
                slabel: Optional[str] = Body(None),
                tlabel: Optional[str] = Body(None)):
    job = _ensure_job(job_id)
    team = _ensure_team(job, team_id)
    _ensure_node(team, source, slabel, sx, sy)
    _ensure_node(team, target, tlabel, tx, ty)
    key = (source, target)
    team["links"][key] = team["links"].get(key, 0) + 1
    return {"ok": True, "weight": team["links"][key]}

@router.post("/reset/{job_id}")
def passnet_reset(job_id: str):
    PASS_STATE.pop(job_id, None)
    return {"ok": True}

# ---------------- Stub data ----------------
@router.get("/stub/{job_id}")
def passnet_stub(job_id: str):
    return {
        "job_id": job_id,
        "teams": [
            {"team_id": 0,
             "nodes": [{"id": 11, "label": "11", "x": 10, "y": 30},
                       {"id": 7,  "label": "7",  "x": 40, "y": 50},
                       {"id": 9,  "label": "9",  "x": 70, "y": 40}],
             "links": [{"source": 11, "target": 7, "weight": 3},
                       {"source": 7,  "target": 9, "weight": 2}]},
            {"team_id": 1,
             "nodes": [{"id": 4, "label": "4", "x": 20, "y": 60},
                       {"id": 6, "label": "6", "x": 50, "y": 30}],
             "links": [{"source": 4, "target": 6, "weight": 1}]}
        ]
    }

# --------------- UIs (no f-strings) ---------------
_HTML_LIVE = """<!doctype html>
<meta charset='utf-8'/>
<title>PassNet LIVE {JOB}</title>
<body style="margin:0;background:#0b1824;color:#e8f1f7;font-family:sans-serif;">
<h3 style="padding:12px">Passing Network (LIVE) — job {JOB}</h3>
<svg id="s" width="100%" height="90vh"></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const job = "{JOB}";
const svg = d3.select("#s");
const W = window.innerWidth, H = window.innerHeight*0.9;

function draw(data){
  svg.selectAll("*").remove();
  const teams = data.teams || [];
  const cols = Math.max(1, teams.length), colW = W/cols;
  teams.forEach((team,i)=>{
    const g = svg.append("g").attr("transform","translate("+(i*colW+40)+",40)");
    const xs = team.nodes.map(d=>(d.x||0)), ys = team.nodes.map(d=>(d.y||0));
    const nx = d3.scaleLinear().domain(d3.extent(xs.length?xs:[0,100])).range([0, colW-80]);
    const ny = d3.scaleLinear().domain(d3.extent(ys.length?ys:[0,100])).range([0, H-120]);
    const wmax = d3.max(team.links, d=>d.weight)||1;
    const sw = d3.scaleLinear().domain([1,wmax]).range([1,10]);
    function byId(id){ return team.nodes.find(n=>n.id===id) || {x:0,y:0}; }
    g.selectAll("line").data(team.links).enter().append("line")
      .attr("x1", d=> nx(byId(d.source).x)).attr("y1", d=> ny(byId(d.source).y))
      .attr("x2", d=> nx(byId(d.target).x)).attr("y2", d=> ny(byId(d.target).y))
      .attr("stroke","#a8c7ff").attr("stroke-opacity",0.85).attr("stroke-width", d=> sw(d.weight));
    const nodes = g.selectAll("g.node").data(team.nodes).enter().append("g")
      .attr("class","node").attr("transform", d=>"translate("+nx(d.x)+","+ny(d.y)+")");
    nodes.append("circle").attr("r",18).attr("fill","#162433").attr("stroke","#e8f1f7").attr("stroke-width",1.5);
    nodes.append("text").text(d=>d.label||d.id).attr("dy","0.35em").attr("text-anchor","middle")
      .attr("fill","#e8f1f7").style("font-weight","700");
  });
}

async function tick(){
  const r = await fetch("/passnet/live/{JOB}");
  const j = await r.json();
  draw(j);
}
tick(); setInterval(tick, 1000);
</script>
</body>"""

_HTML_STUB = """<!doctype html>
<meta charset='utf-8'/>
<title>PassNet {JOB}</title>
<body style="margin:0;background:#0b1824;color:#e8f1f7;font-family:sans-serif;">
<h3 style="padding:12px">Passing Network — job {JOB}</h3>
<svg id="s" width="100%" height="90vh"></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const job = "{JOB}";
async function tick(){
  const r = await fetch("/passnet/stub/{JOB}");
  const j = await r.json();
  document.body.querySelector("#s").innerHTML = ""; // minimal
}
tick(); setInterval(tick, 1000);
</script>
</body>"""

@router.get("/ui-live/{job_id}")
def passnet_ui_live(job_id: str):
    return HTMLResponse(_HTML_LIVE.replace("{JOB}", job_id))

@router.get("/ui/{job_id}")
def passnet_ui(job_id: str):
    return HTMLResponse(_HTML_STUB.replace("{JOB}", job_id))