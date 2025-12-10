import json
import numpy as np
import os
import math
from pathlib import Path
from collections import Counter

BASE = Path("runs/json")
TRACKS = BASE / "tracks.json"
EVENTS = BASE / "final_events_viewer.json"
FORM = BASE / "formation.json"
CHAINS = BASE / "possession_chains.json"
TACTICS = BASE / "advanced_metrics.json"

def main():
    print("\n🔍 STARTING FULL SYSTEM FORENSIC AUDIT\n" + "="*40)

    # 1. TRACKING & PHYSICS
    print("\n[1] TRACKING & PHYSICS ENGINE")
    if not TRACKS.exists(): print("❌ Tracks file missing!"); return
    
    try:
        data = json.loads(TRACKS.read_text())
        frames = data['frames']
        print(f"   🎥 Total Frames: {len(frames)} ({len(frames)/25:.1f} seconds)")
        
        # Physics Check
        has_meters = 0
        speeds = []
        ids = set()
        ball_frames = 0
        
        for f in frames:
            if f.get('ball'): ball_frames += 1
            for p in f['players']:
                ids.add(p['id'])
                if p.get('x_m', -1) != -1: has_meters += 1
                if p.get('speed', 0) > 0: speeds.append(p['speed'])
        
        print(f"   👥 Unique IDs: {len(ids)} (Target: <50 for short clips, <200 for full match)")
        print(f"   ⚽ Ball Detected: {ball_frames} frames ({ball_frames/len(frames):.1%})")
        
        if has_meters > 0:
            print(f"   📐 Calibration: ✅ ACTIVE ({has_meters} player-frames have meters)")
            if speeds:
                print(f"   🏃 Max Speed: {max(speeds):.1f} km/h (Target: 25-36)")
                print(f"   🏃 Avg Speed: {sum(speeds)/len(speeds):.1f} km/h (Target: 4-8)")
            else:
                print("   ❌ Physics calculated but NO SPEED found.")
        else:
            print("   ❌ Calibration FAILED (All coordinates are -1)")
            
    except Exception as e: print(f"   ❌ Error parsing tracks: {e}")

    # 2. FORMATION & TEAMS
    print("\n[2] TEAMS & TACTICS")
    if not FORM.exists(): print("❌ Formation file missing!")
    else:
        try:
            form = json.loads(FORM.read_text())
            print(f"   🛡️ Home Formation: {form.get('A', 'Unknown')}")
            print(f"   ⚔️ Away Formation: {form.get('B', 'Unknown')}")
            # Check for the "100-0" bug
            if "0-0-0" in str(form) or "Unknown" in str(form):
                 print("   ⚠️ WARNING: Formation logic struggled (Low data or bad stitching).")
            else:
                 print("   ✅ Formations look valid.")
        except: print("   ❌ Corrupt formation file.")

    # 3. EVENTS & INTELLIGENCE
    print("\n[3] EVENT BRAIN (LSTM)")
    if not EVENTS.exists(): print("❌ Events file missing!")
    else:
        try:
            events = json.loads(EVENTS.read_text())
            if not events:
                print("   ⚠️ No events found.")
            else:
                counts = Counter([e['label'] for e in events])
                print(f"   🧠 Total Clips: {len(events)}")
                print(f"   📊 Breakdown: {dict(counts)}")
                
                # Check for Teams
                unknown_teams = sum(1 for e in events if e.get('team') == 'Unknown')
                if unknown_teams == len(events):
                    print("   ❌ CRITICAL: All events have 'Unknown' team.")
                else:
                    print(f"   ✅ Team Assignment: {len(events)-unknown_teams}/{len(events)} events have teams.")
                
                # Check for Value
                has_value = sum(1 for e in events if e.get('xt', 0) > 0 or e.get('xg', 0) > 0)
                if has_value > 0:
                    print(f"   💰 Value Engine: ✅ ONLINE ({has_value} events have xG/xT)")
                else:
                    print("   ❌ Value Engine: OFFLINE (All xG/xT are 0)")
        except: print("   ❌ Corrupt events file.")

    # 4. POSSESSION
    print("\n[4] POSSESSION ENGINE")
    if not CHAINS.exists(): print("❌ Chains file missing!")
    else:
        try:
            chains = json.loads(CHAINS.read_text())
            print(f"   🔗 Total Chains: {len(chains)}")
            if len(chains) > 0:
                phases = Counter([c.get('phase', 'Unknown') for c in chains])
                print(f"   📊 Phases: {dict(phases)}")
                if phases.get('Unknown', 0) == len(chains):
                    print("   ❌ Tactical Phases missing (Physics link broken).")
                else:
                    print("   ✅ Tactical Phases Detected.")
        except: print("   ❌ Corrupt chains file.")

    print("\n" + "="*40)

if __name__ == "__main__":
    main()
