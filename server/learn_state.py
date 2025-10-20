# server/learn_state.py
import os, json, time
STATE_PATH = os.path.abspath("./files/calibration.json")

DEFAULTS = {
    "m_per_px": 0.25,
    "exclude_top_pct": 0.25,
    "min_overlap": 0.60,
    "conf_min": 0.60,
    "updated_at": 0
}

def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
            return {**DEFAULTS, **s}
    except Exception:
        return DEFAULTS.copy()

def save_state(**kwargs):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    s = load_state()
    s.update(kwargs)
    s["updated_at"] = int(time.time())
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    return s
