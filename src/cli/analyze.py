from __future__ import annotations
import argparse, json, logging, os, shutil, subprocess, sys, time
from pathlib import Path
from typing import Sequence

LOGGER_NAME = "soccer.cli.analyze"
class PipelineError(RuntimeError): ...

def _repo_root() -> Path: return Path(__file__).resolve().parents[2]
def _script(n:str)->Path:
    p=_repo_root()/n
    if not p.exists(): raise PipelineError(f"Required script missing: {n}")
    return p

def _log(path:Path, verbose=False):
    lg=logging.getLogger(LOGGER_NAME); lg.setLevel(logging.DEBUG if verbose else logging.INFO); lg.handlers.clear()
    fmt=logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh=logging.FileHandler(path, encoding="utf-8"); fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); lg.addHandler(fh)
    ch=logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.setLevel(logging.DEBUG if verbose else logging.INFO); lg.addHandler(ch)
    return lg

def _run(lg, step, cmd, env=None):
    lg.info("[%s] starting", step)
    t0=time.perf_counter()
    p=subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for ln in (p.stdout or "").splitlines(): lg.info("[%s] %s", step, ln)
    if p.returncode!=0: raise PipelineError(f"[{step}] exited {p.returncode}")
    lg.info("[%s] finished in %.2fs", step, time.perf_counter()-t0)

def _ensure_H(lg, out_dir:Path, src:Path|None)->Path:
    dst=out_dir/"homography.json"
    if dst.exists(): return dst
    if not src or not src.exists(): raise PipelineError(f"No homography. Provide --homography or place {dst}")
    shutil.copy2(src,dst); lg.info("Copied homography %s -> %s", src, dst); return dst

def main(argv: Sequence[str] | None = None) -> int:
    ap=argparse.ArgumentParser("One-command soccer analysis")
    ap.add_argument("--video", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=25.0); ap.add_argument("--resize_w", type=int, default=1280)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--homography", default=None)
    ap.add_argument("--no_passes", action="store_true"); ap.add_argument("--no_maps", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a=ap.parse_args(argv)

    out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    lg=_log(out/"run.log", a.verbose)
    lg.info("Args: %s", vars(a))
    vid=Path(a.video)
    if not vid.exists(): lg.error("Video not found: %s", vid); return 2

    (out/"run_manifest.json").write_text(json.dumps({
        "video": str(vid.resolve()), "fps": a.fps, "resize_w": a.resize_w, "max_frames": a.max_frames,
        "homography": a.homography, "skip_passes": bool(a.no_passes), "skip_maps": bool(a.no_maps),
        "timestamp": time.time(), "schema_version": "v1.0"
    }, indent=2), encoding="utf-8")

    try:
        env=os.environ.copy()
        env["PYTHONPATH"]=str(_repo_root()) + os.pathsep + env.get("PYTHONPATH","")

        # 1) detect + track
        cmd1=[sys.executable, str(_script("run_soccer.py")),
              "--source_video_path", str(vid),
              "--target_video_path", str(out/"out_ids.mp4"),
              "--export_json",       str(out/"tracks.json"),
              "--resize_w",          str(a.resize_w),
              "--detect_ball"]
        if a.max_frames>0: cmd1+=["--max_frames", str(a.max_frames)]
        _run(lg, "detect_track", cmd1, env)

        # 2) team split
        _run(lg, "team_split", [sys.executable, str(_script("quick_team_split.py")),
            "--tracks", str(out/"tracks.json"),
            "--out",    str(out/"teams.json")], env)

        # 3) homography
        H=_ensure_H(lg, out, Path(a.homography) if a.homography else None)

        # 4) meters + speeds
        _run(lg, "project_to_meters", [sys.executable, str(_script("project_tracks_to_meters.py")),
            "--tracks",     str(out/"teams.json"),
            "--H",          str(H),
            "--out_tracks", str(out/"teams_metric.json"),
            "--out_speeds", str(out/"speeds_ms.csv")], env)

        if not a.no_passes:
            # 5) passes
            _run(lg, "passes", [sys.executable, str(_script("make_passnet_meters_v2.py")),
                "--teams_metric", str(out/"teams_metric.json"),
                "--H",            str(H),
                "--out_json",     str(out/"passes.json"),
                "--out_csv",      str(out/"passnet.csv")], env)

            if not a.no_maps:
                # 6) pass maps
                for team in (0,1):
                    _run(lg, f"passmap_team{team}", [sys.executable, str(_script("passmap_pro.py")),
                        "--teams_metric", str(out/"teams_metric.json"),
                        "--passes",       str(out/"passes.json"),
                        "--team",         str(team),
                        "--time",         "all",
                        "--min_frames",   "80",
                        "--min_passes",   "1",
                        "--top_edges",    "50",
                        "--fold_half",    "--nodes_from_passes",
                        "--out",          str(out/f"passmap_team{team}.png")], env)
        else:
            lg.info("Skipping passes/maps")

        lg.info("Pipeline finished successfully"); return 0
    except PipelineError as e:
        lg.error(str(e)); lg.info("Pipeline failed"); return 1
    except Exception:
        import traceback; traceback.print_exc(); lg.info("Pipeline failed (unexpected)"); return 1

if __name__=="__main__":
    raise SystemExit(main())
