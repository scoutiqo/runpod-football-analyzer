import json
import numpy as np
from pathlib import Path
from collections import Counter

TRACKS_FILE = "runs/json/tracks.json"
EVENTS_FILE = "runs/json/final_events_viewer.json"
CHAINS_FILE = "runs/json/possession_chains.json"

def audit_tracking():
    print("\n🔍 AUDITING TRACKING QUALITY...")
    if not Path(TRACKS_FILE).exists():
        print("   ❌ Tracks file missing.")
        return

    data = json.loads(Path(TRACKS_FILE).read_text())
    frames = data.get('frames', [])
    
    id_lifespans = {}
    total_frames = len(frames)
    
    for f in frames:
        for p in f.get('players', []):
            pid = str(p['id'])
            id_lifespans[pid] = id_lifespans.get(pid, 0) + 1
            
    lifespans = list(id_lifespans.values())
    if not lifespans:
        print("   ❌ No players found.")
        return

    avg_life_frames = np.mean(lifespans)
    avg_life_sec = avg_life_frames / 25.0
    total_ids = len(id_lifespans)
    short_tracks = sum(1 for l in lifespans if l < 25) # Less than 1 second
    
    print(f"   📊 Total Unique IDs: {total_ids} (Ideal: ~30-40 for a match)")
    print(f"   ⏱️ Average Track Duration: {avg_life_sec:.2f} seconds")
    print(f"   👻 Ghost Tracks (<1s): {short_tracks} ({round(short_tracks/total_ids*100)}%)")
    
    if avg_life_sec < 5.0:
        print("   🚨 DIAGNOSIS: SEVERE FLICKERING. Players are losing IDs constantly.")
    else:
        print("   ✅ DIAGNOSIS: Tracking is stable.")

def audit_events_and_values():
    print("\n🔍 AUDITING EVENTS & VALUE METRICS...")
    if not Path(EVENTS_FILE).exists():
        print("   ❌ Events file missing.")
        return

    events = json.loads(Path(EVENTS_FILE).read_text())
    
    # Check for xG / xThreat
    has_xg = any('xg' in e for e in events)
    has_xt = any('xt' in e for e in events)
    
    labels = Counter([e['label'] for e in events])
    
    print(f"   📊 Total Events: {len(events)}")
    print(f"   🏷️  Event Types Found: {list(labels.keys())}")
    
    if not has_xg:
        print("   ❌ xG (Expected Goals) is MISSING.")
    else:
        print("   ✅ xG detected.")
        
    if not has_xt:
        print("   ❌ xT (Expected Threat) is MISSING.")
    else:
        print("   ✅ xT detected.")

def audit_possession():
    print("\n🔍 AUDITING POSSESSION CHAINS...")
    if not Path(CHAINS_FILE).exists():
        print("   ❌ Chains file missing.")
        return

    chains = json.loads(Path(CHAINS_FILE).read_text())
    print(f"   🔗 Total Possession Chains: {len(chains)}")
    
    if len(chains) == 0:
        print("   🚨 DIAGNOSIS: Possession Engine failed to link events.")
        return
        
    avg_passes = np.mean([c.get('pass_count', 0) for c in chains])
    print(f"   ⚽ Avg Passes per Chain: {avg_passes:.1f}")

if __name__ == "__main__":
    audit_tracking()
    audit_events_and_values()
    audit_possession()
