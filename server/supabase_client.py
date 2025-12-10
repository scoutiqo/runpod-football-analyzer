import os
from typing import List, Dict

try:
    from supabase import create_client
except Exception:
    create_client = None

# simple in-process cache to avoid repeated DB hits in one run
_CACHE: dict[str, list[dict]] = {}

def _get_client():
    """
    Server-side only. Do NOT import this in browser code.
    Uses SUPABASE_URL + SUPABASE_KEY (anon/service) from environment.
    """
    url = os.getenv("SUPABASE_URL")
    # prefer anon key; service role ONLY on backend hosts
    key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key and create_client):
        return None
    try:
        return create_client(url.rstrip("/"), key)
    except Exception:
        return None

def fetch_teams_cached() -> List[Dict]:
    if "teams" in _CACHE:
        return _CACHE["teams"]

    sb = _get_client()
    if not sb:
        out = [{"id": "A", "name": "TEAM A"}, {"id": "B", "name": "TEAM B"}]
        _CACHE["teams"] = out
        return out

    candidates = [
        ("teams", "id,name"),
        ("cached_teams", "id,name"),
        ("scout_cache_teams", "id,name"),
    ]
    for table, cols in candidates:
        try:
            res = sb.table(table).select(cols).limit(50).execute()
            data = res.data or []
            if data:
                out = [{"id": str(x.get("id")), "name": x.get("name") or f"Team {x.get('id')}"} for x in data]
                _CACHE["teams"] = out
                return out
        except Exception:
            continue

    out = [{"id": "A", "name": "TEAM A"}, {"id": "B", "name": "TEAM B"}]
    _CACHE["teams"] = out
    return out

def fetch_roster(team_id: str) -> List[Dict]:
    cache_key = f"roster:{team_id}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    sb = _get_client()
    if not sb:
        return []

    candidates = [
        ("players", "id,name,number", "team_id"),
        ("roster", "id,name,number,team_id", "team_id"),
        ("cached_players", "id,name,number,team_id", "team_id"),
    ]
    for table, cols, team_col in candidates:
        try:
            res = sb.table(table).select(cols).eq(team_col, team_id).limit(40).execute()
            arr = res.data or []
            if arr:
                out = [{
                    "id": str(x.get("id")),
                    "name": x.get("name") or f"Player {x.get('id')}",
                    "number": x.get("number"),
                } for x in arr]
                _CACHE[cache_key] = out
                return out
        except Exception:
            continue
    return []
