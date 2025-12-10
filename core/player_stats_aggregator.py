import json
from pathlib import Path
from collections import defaultdict

CHAINS_FILE = "runs/json/possession_chains.json"
EVENTS_FILE = "runs/json/final_events_viewer.json"
OUTPUT_STATS = "runs/json/player_stats_db.json"

def main():
    print("📊 Aggregating PLAYER STATS (Success Rates, Totals)...")
    
    if not Path(CHAINS_FILE).exists(): return
    chains = json.loads(Path(CHAINS_FILE).read_text())
    
    # Dictionary: PlayerID -> Stats
    player_db = defaultdict(lambda: {
        "passes_total": 0, "passes_completed": 0,
        "duels_total": 0, "duels_won": 0,
        "shots_total": 0, "shots_on_target": 0,
        "distance_m": 0.0, "sprints": 0,
        "tactical_actions": []
    })
    
    for chain in chains:
        events = chain['events']
        # If chain ended nicely (Pass/Shot), the previous passes were successful
        # If chain ended in 'Ball Loss', the last action failed.
        
        chain_failed = "ball_loss" in [e['label'].lower() for e in events]
        
        for i, evt in enumerate(events):
            pid = str(evt.get('actor_id', 'unknown'))
            if pid == 'unknown': continue
            
            label = evt['label'].lower()
            
            # PASSING
            if "pass" in label or "cross" in label:
                player_db[pid]['passes_total'] += 1
                # Success if it wasn't the last event of a failed chain
                is_last = (i == len(events) - 1)
                if not (is_last and chain_failed):
                    player_db[pid]['passes_completed'] += 1
            
            # DUELS (Tackles/Aerials)
            if "duel" in label or "tackle" in label:
                player_db[pid]['duels_total'] += 1
                # Success if their team KEPT the chain after this event
                if not chain_failed:
                    player_db[pid]['duels_won'] += 1

    # CALCULATE PERCENTAGES
    final_db = []
    for pid, stats in player_db.items():
        # Pass %
        if stats['passes_total'] > 0:
            stats['pass_completion_pct'] = round((stats['passes_completed'] / stats['passes_total']) * 100, 1)
        else:
            stats['pass_completion_pct'] = 0.0
            
        # Duel Win %
        if stats['duels_total'] > 0:
            stats['duel_win_pct'] = round((stats['duels_won'] / stats['duels_total']) * 100, 1)
        else:
            stats['duel_win_pct'] = 0.0
            
        final_db.append({"player_id": pid, "stats": stats})
        
    Path(OUTPUT_STATS).write_text(json.dumps(final_db, indent=2))
    print(f"✅ Calculated Stats for {len(final_db)} players.")

if __name__ == "__main__":
    main()

